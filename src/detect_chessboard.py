"""
Testa detecção do tabuleiro nas câmeras ou em uma foto.

Uso:
  python src/detect_chessboard.py
  python src/detect_chessboard.py --image images/stereoLeft/imageL0.png
  python src/detect_chessboard.py --size 7 7
  python src/detect_chessboard.py --auto        # detecta 8x8 ou 8x4 sozinho + erro %
  python src/detect_chessboard.py --board-width 34.624 --board-height 34.624

Comportamento:
  - Os valores calculados por quadro (padrão e erro %) são ACUMULADOS ao
    longo das leituras para estimar a média mais correta;
  - Tecla 'w' ou 'W' grava os parâmetros de reconhecimento (padrão de cantos
    e, se informado, dimensões do tabuleiro) no calibration.json, exibindo no
    console os dados calculados e a diferença vs. a configuração anterior;
  - Tecla ESC (ou 'q') sai SEM gravar — a configuração anterior é mantida.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2

SRC = Path(__file__).resolve().parent
PROJECT_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chessboard_auto import board_error_pct, detect_chessboard_auto, square_size_cm

CALIBRATION_JSON = PROJECT_ROOT / "calibration.json"

FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

SAVE_KEYS = (ord("w"), ord("W"))
QUIT_KEYS = (27, ord("q"), ord("Q"))


@dataclass
class DetectionStats:
    """Acumula as leituras de reconhecimento para estimar valores médios."""

    configured: tuple[int, int] | None = None
    frames: int = 0
    detections: int = 0
    errors: list[float] = field(default_factory=list)
    pattern_counts: dict[tuple[int, int], int] = field(default_factory=dict)

    def record(self, size: tuple[int, int] | None, err: float | None) -> None:
        """Registra uma leitura de um quadro (size=None = não detectado)."""
        self.frames += 1
        if size is None:
            return
        self.detections += 1
        key = tuple(size)
        self.pattern_counts[key] = self.pattern_counts.get(key, 0) + 1
        if err is not None and math.isfinite(err):
            self.errors.append(err)

    @property
    def mean_error(self) -> float | None:
        return sum(self.errors) / len(self.errors) if self.errors else None

    @property
    def min_error(self) -> float | None:
        return min(self.errors) if self.errors else None

    @property
    def max_error(self) -> float | None:
        return max(self.errors) if self.errors else None

    @property
    def dominant_pattern(self) -> tuple[int, int] | None:
        """Padrão mais frequente; sem detecções, volta ao configurado."""
        if not self.pattern_counts:
            return self.configured
        return max(self.pattern_counts, key=self.pattern_counts.get)

    def summary(self) -> str:
        pct = 100.0 * self.detections / self.frames if self.frames else 0.0
        lines = [
            f"Frames analisados: {self.frames} | detecções: {self.detections} ({pct:.1f}%)"
        ]
        if self.pattern_counts:
            pat = ", ".join(
                f"{c + 1}x{r + 1} x{n}"
                for (c, r), n in sorted(self.pattern_counts.items())
            )
            lines.append(f"Padrões detectados: {pat}")
        if self.errors:
            lines.append(
                f"Erro médio: {self.mean_error:.2f}% "
                f"(min {self.min_error:.2f}%, máx {self.max_error:.2f}%)"
            )
        return "\n".join(lines)


def load_calibration(path: Path = CALIBRATION_JSON) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_recognition_params(
    size: tuple[int, int],
    board_size_cm: tuple[float, float] | None = None,
    calibration_error_pct: float | None = None,
    path: Path = CALIBRATION_JSON,
) -> dict:
    """Salva no calibration.json os parâmetros usados no reconhecimento.

    Atualiza chessboardSize/pattern (e board_size_cm/square_cm quando
    informado, e calibration_error_pct quando calculado), preservando o
    restante do arquivo (Q, baseline, câmeras...).
    """
    data = load_calibration(path)
    cols, rows = size
    data["chessboardSize"] = [cols, rows]
    data["pattern"] = f"{cols + 1}x{rows + 1}"
    if board_size_cm is not None:
        w, h = float(board_size_cm[0]), float(board_size_cm[1])
        data["board_size_cm"] = [w, h]
        data["square_cm"] = round(square_size_cm(w, h, size), 3)
    if calibration_error_pct is not None:
        data["calibration_error_pct"] = round(calibration_error_pct, 2)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return data


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, list):
        return "[" + ", ".join(str(x) for x in value) + "]"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def build_diff_text(
    previous: dict,
    size: tuple[int, int],
    board_size_cm: tuple[float, float] | None = None,
    mean_error: float | None = None,
) -> str:
    """Diferença entre a configuração anterior (calibration.json) e a proposta."""
    cols, rows = size
    pattern = f"{cols + 1}x{rows + 1}"
    proposals = [
        ("chessboardSize", previous.get("chessboardSize"), [cols, rows]),
        ("pattern", previous.get("pattern"), pattern),
    ]
    if board_size_cm is not None:
        w, h = float(board_size_cm[0]), float(board_size_cm[1])
        proposals.append(("board_size_cm", previous.get("board_size_cm"), [w, h]))
        proposals.append(
            ("square_cm", previous.get("square_cm"), round(square_size_cm(w, h, size), 3))
        )
    if mean_error is not None:
        proposals.append(
            ("calibration_error_pct", previous.get("calibration_error_pct"), round(mean_error, 2))
        )

    lines = []
    for name, old, new in proposals:
        if old == new:
            lines.append(f"  {name}: {_fmt(new)} (inalterado)")
        else:
            lines.append(f"  {name}: {_fmt(old)} -> {_fmt(new)}")
    return "\n".join(lines) if lines else "  (nenhum campo proposto)"


def detect(gray: cv2.Mat, size: tuple[int, int]) -> tuple[bool, object]:
    found, corners = cv2.findChessboardCorners(gray, size, FIND_FLAGS)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    return found, corners


def draw_result(frame, size: tuple[int, int], found: bool, corners) -> None:
    vis = frame.copy()
    if found:
        cv2.drawChessboardCorners(vis, size, corners, found)
        cv2.putText(vis, f"OK {size[0]}x{size[1]} cantos", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(vis, "nao detectado", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return vis


def test_image(path: Path, size: tuple[int, int]) -> tuple[DetectionStats, int]:
    """Testa uma foto; retorna (estatísticas acumuladas, tecla pressionada)."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"Erro ao ler {path}")
        return DetectionStats(configured=size), 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = detect(gray, size)
    err = board_error_pct(gray, corners, size) if found else None
    stats = DetectionStats(configured=size)
    stats.record(size if found else None, err)
    print(f"{'OK' if found else 'FALHOU'}: {path.name}  size={size}  erro {err:.2f}%"
          if found else f"FALHOU: {path.name}  size={size}")
    cv2.imshow("detect_chessboard", draw_result(img, size, found, corners))
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    return stats, key


def test_live(
    size: tuple[int, int],
    on_save: Callable[[DetectionStats], None] | None = None,
    on_abandon: Callable[[DetectionStats], None] | None = None,
) -> None:
    cap0 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    for cap in (cap0, cap2):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    stats = DetectionStats(configured=size)
    print(
        f"Cantos internos: {size[0]}x{size[1]}. "
        f"'w'=gravar | ESC/q=sair sem gravar"
    )
    while True:
        ok0, f0 = cap0.read()
        ok2, f2 = cap2.read()
        if not ok0 or not ok2:
            break
        for frame, title in ((f0, "video0"), (f2, "video2")):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = detect(gray, size)
            err = board_error_pct(gray, corners, size) if found else None
            stats.record(size if found else None, err)
            cv2.imshow(title, draw_result(frame, size, found, corners))
        key = cv2.waitKey(1) & 0xFF
        if key in SAVE_KEYS and on_save is not None:
            on_save(stats)
        elif key in QUIT_KEYS:
            break
    cap0.release()
    cap2.release()
    cv2.destroyAllWindows()
    if on_abandon is not None:
        on_abandon(stats)


def test_image_auto(path: Path) -> tuple[DetectionStats, int]:
    """Detecta 8x8 ou 8x4 automaticamente em uma foto.

    Retorna (estatísticas acumuladas, tecla pressionada).
    """
    img = cv2.imread(str(path))
    if img is None:
        print(f"Erro ao ler {path}")
        return DetectionStats(), 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = detect_chessboard_auto(gray)
    stats = DetectionStats()
    if result is None:
        stats.record(None, None)
        print(f"FALHOU: {path.name}  (nenhum padrão 8x8/8x4 detectado)")
        cv2.putText(img, "nao detectado", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        size, corners = result
        err = board_error_pct(gray, corners, size)
        stats.record(size, err)
        cols, rows = size
        cv2.drawChessboardCorners(img, size, corners, True)
        cv2.putText(img, f"OK {cols + 1}x{rows + 1} | erro {err:.2f}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f"OK: {path.name}  tabuleiro {cols + 1}x{rows + 1}  erro {err:.2f}%")
    cv2.imshow("detect_chessboard --auto", img)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    return stats, key


def test_live_auto(
    on_save: Callable[[DetectionStats], None] | None = None,
    on_abandon: Callable[[DetectionStats], None] | None = None,
) -> None:
    """Loop ao vivo: detecta o padrão sozinho e acumula erro % por quadro."""
    cap0 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    for cap in (cap0, cap2):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    stats = DetectionStats()
    print("Auto: 8x8 ou 8x4. 'w'=gravar | ESC/q=sair sem gravar")
    while True:
        ok0, f0 = cap0.read()
        ok2, f2 = cap2.read()
        if not ok0 or not ok2:
            break
        for frame, title in ((f0, "video0"), (f2, "video2")):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = detect_chessboard_auto(gray)
            if result is None:
                stats.record(None, None)
                cv2.putText(frame, "nao detectado", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                size, corners = result
                err = board_error_pct(gray, corners, size)
                stats.record(size, err)
                cols, rows = size
                cv2.drawChessboardCorners(frame, size, corners, True)
                cv2.putText(frame, f"{cols + 1}x{rows + 1} | erro {err:.2f}%", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in SAVE_KEYS and on_save is not None:
            on_save(stats)
        elif key in QUIT_KEYS:
            break
    cap0.release()
    cap2.release()
    cv2.destroyAllWindows()
    if on_abandon is not None:
        on_abandon(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--size", type=int, nargs=2, default=[7, 7],
                       metavar=("COLS", "ROWS"), help="cantos internos (padrão 7 7)")
    group.add_argument("--auto", action="store_true",
                       help="detecta 8x8 ou 8x4 sozinho e mostra o erro percentual")
    parser.add_argument("--board-width", type=float,
                        help="largura do tabuleiro em cm (salva no calibration.json)")
    parser.add_argument("--board-height", type=float,
                        help="altura do tabuleiro em cm (salva no calibration.json)")
    args = parser.parse_args()

    if (args.board_width is None) != (args.board_height is None):
        parser.error("--board-width e --board-height devem ser informados juntos")
    board_size = None
    if args.board_width is not None:
        board_size = (args.board_width, args.board_height)

    size = (args.size[0], args.size[1])

    def salvar(stats: DetectionStats) -> None:
        detected = stats.dominant_pattern
        if detected is None:
            print("✘ Nenhum padrão detectado — nada a salvar (configuração anterior mantida).")
            return
        previous = load_calibration()
        mean = stats.mean_error
        print("\n--- Dados calculados (acumulados até agora) ---")
        print(stats.summary())
        print("--- Diferença vs. configuração anterior ---")
        print(build_diff_text(previous, detected, board_size_cm=board_size, mean_error=mean))
        save_recognition_params(detected, board_size_cm=board_size, calibration_error_pct=mean)
        print(f"✔ Dados GRAVADOS — configuração atualizada em {CALIBRATION_JSON}.\n")

    def abandonar(stats: DetectionStats) -> None:
        print("\n--- Dados calculados (acumulados até agora) ---")
        print(stats.summary())
        detected = stats.dominant_pattern
        if detected is not None:
            previous = load_calibration()
            print("--- Diferença que seria aplicada vs. configuração anterior ---")
            print(build_diff_text(previous, detected, board_size_cm=board_size,
                                  mean_error=stats.mean_error))
        print("✘ Dados ABANDONADOS — configuração anterior mantida (nada gravado).\n")

    if args.image:
        if args.auto:
            stats, key = test_image_auto(args.image)
        else:
            stats, key = test_image(args.image, size)
        if key in SAVE_KEYS:
            salvar(stats)
        else:
            abandonar(stats)
    else:
        if args.auto:
            test_live_auto(on_save=salvar, on_abandon=abandonar)
        else:
            test_live(size, on_save=salvar, on_abandon=abandonar)


if __name__ == "__main__":
    main()
