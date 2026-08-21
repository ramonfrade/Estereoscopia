"""
Testa detecção do tabuleiro nas câmeras ou em uma foto.

Uso:
  python src/detect_chessboard.py
  python src/detect_chessboard.py --image images/stereoLeft/imageL0.png
  python src/detect_chessboard.py --size 7 7
  python src/detect_chessboard.py --auto        # detecta 8x8 ou 8x4 sozinho + erro %
  python src/detect_chessboard.py --board-width 34.624 --board-height 34.624

Os parâmetros usados no reconhecimento (padrão de cantos e, se informado,
dimensões do tabuleiro) são salvos em calibration.json (mesmo arquivo da
calibração), preservando os demais campos.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
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


def save_recognition_params(
    size: tuple[int, int],
    board_size_cm: tuple[float, float] | None = None,
    path: Path = CALIBRATION_JSON,
) -> dict:
    """Salva no calibration.json os parâmetros usados no reconhecimento.

    Atualiza chessboardSize/pattern (e board_size_cm/square_cm quando
    informado), preservando o restante do arquivo (Q, baseline, câmeras...).
    """
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    cols, rows = size
    data["chessboardSize"] = [cols, rows]
    data["pattern"] = f"{cols + 1}x{rows + 1}"
    if board_size_cm is not None:
        w, h = float(board_size_cm[0]), float(board_size_cm[1])
        data["board_size_cm"] = [w, h]
        data["square_cm"] = round(square_size_cm(w, h, size), 3)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return data


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


def test_image(path: Path, size: tuple[int, int]) -> bool:
    img = cv2.imread(str(path))
    if img is None:
        print(f"Erro ao ler {path}")
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = detect(gray, size)
    print(f"{'OK' if found else 'FALHOU'}: {path.name}  size={size}")
    cv2.imshow("detect_chessboard", draw_result(img, size, found, corners))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return found


def test_live(size: tuple[int, int]) -> None:
    cap0 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    for cap in (cap0, cap2):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Cantos internos: {size[0]}x{size[1]}. ESC=sair")
    while True:
        ok0, f0 = cap0.read()
        ok2, f2 = cap2.read()
        if not ok0 or not ok2:
            break
        for frame, title in ((f0, "video0"), (f2, "video2")):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = detect(gray, size)
            cv2.imshow(title, draw_result(frame, size, found, corners))
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap0.release()
    cap2.release()
    cv2.destroyAllWindows()


def test_image_auto(path: Path):
    """Detecta 8x8 ou 8x4 automaticamente em uma foto e mostra o erro %.

    Retorna (size, corners) quando detectado, senão None.
    """
    img = cv2.imread(str(path))
    if img is None:
        print(f"Erro ao ler {path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = detect_chessboard_auto(gray)
    if result is None:
        print(f"FALHOU: {path.name}  (nenhum padrão 8x8/8x4 detectado)")
        cv2.putText(img, "nao detectado", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        size, corners = result
        err = board_error_pct(gray, corners, size)
        cols, rows = size
        cv2.drawChessboardCorners(img, size, corners, True)
        cv2.putText(img, f"OK {cols + 1}x{rows + 1} | erro {err:.2f}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f"OK: {path.name}  tabuleiro {cols + 1}x{rows + 1}  erro {err:.2f}%")
    cv2.imshow("detect_chessboard --auto", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result


def test_live_auto(on_detect: Callable[[tuple[int, int]], None] | None = None) -> None:
    """Loop ao vivo: detecta o padrão sozinho e mostra o erro % continuamente.

    on_detect é chamado uma única vez, no primeiro padrão detectado.
    """
    cap0 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    for cap in (cap0, cap2):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Auto: 8x8 ou 8x4. ESC=sair")
    saved = False
    while True:
        ok0, f0 = cap0.read()
        ok2, f2 = cap2.read()
        if not ok0 or not ok2:
            break
        for frame, title in ((f0, "video0"), (f2, "video2")):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = detect_chessboard_auto(gray)
            if result is None:
                cv2.putText(frame, "nao detectado", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                size, corners = result
                if on_detect is not None and not saved:
                    on_detect(size)
                    saved = True
                err = board_error_pct(gray, corners, size)
                cols, rows = size
                cv2.drawChessboardCorners(frame, size, corners, True)
                cv2.putText(frame, f"{cols + 1}x{rows + 1} | erro {err:.2f}%", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(title, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap0.release()
    cap2.release()
    cv2.destroyAllWindows()


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

    def salvar(detected: tuple[int, int]) -> None:
        save_recognition_params(detected, board_size_cm=board_size)
        print(
            f"Parâmetros salvos em {CALIBRATION_JSON}: "
            f"canto {detected[0]}x{detected[1]} | padrão {detected[0] + 1}x{detected[1] + 1}"
        )

    if args.image:
        if args.auto:
            result = test_image_auto(args.image)
            if result is not None:
                salvar(result[0])
        else:
            test_image(args.image, size)
            salvar(size)
    else:
        if args.auto:
            test_live_auto(on_detect=salvar)
        else:
            test_live(size)
            salvar(size)


if __name__ == "__main__":
    main()
