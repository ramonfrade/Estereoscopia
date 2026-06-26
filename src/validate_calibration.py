"""
Valida a calibração estéreo: retificação lado a lado e métricas.

Uso (na raiz do projeto):
  python src/validate_calibration.py
  python src/validate_calibration.py --pair 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
PROJECT_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import calibration


def row_correlation(rect_l: np.ndarray, rect_r: np.ndarray) -> float:
    g_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    g_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    mask = (g_l > 15) & (g_r > 15)
    if not np.any(mask):
        return float("nan")
    y = int(np.median(np.where(mask)[0]))
    row_l, row_r = g_l[y], g_r[y]
    m = (row_l > 15) & (row_r > 15)
    if m.sum() < 50:
        return float("nan")
    return float(np.corrcoef(row_l[m].astype(float), row_r[m].astype(float))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validar calibração estéreo")
    parser.add_argument("--pair", type=int, default=0, help="Índice imageL0/imageR0, etc.")
    args = parser.parse_args()

    calib_path = PROJECT_ROOT / "calibration.json"
    if calib_path.exists():
        data = json.loads(calib_path.read_text(encoding="utf-8"))
        rms = data.get("stereo_rms_px")
        pairs = data.get("valid_pairs")
        baseline = data.get("baseline_units")
        phys = data.get("physical_baseline_cm")
        print("--- calibration.json ---")
        if rms is not None:
            print(f"  RMS estéreo: {rms:.3f} px  (bom < 1.0, aceitável < 2.0)")
        if pairs is not None:
            print(f"  Pares válidos: {pairs}")
        if baseline is not None and phys is not None:
            print(f"  Baseline calib: {baseline:.2f} | físico: {phys} cm")
    else:
        print("AVISO: calibration.json não encontrado. Rode stereo_calibration.py primeiro.")

    map_path = PROJECT_ROOT / "stereoMap.xml"
    if not map_path.exists():
        raise SystemExit(f"stereoMap.xml não encontrado em {map_path}")

    left_dir = PROJECT_ROOT / "images" / "stereoLeft"
    right_dir = PROJECT_ROOT / "images" / "stereoRight"
    left = left_dir / f"imageL{args.pair}.png"
    right = right_dir / f"imageR{args.pair}.png"

    if not left.exists():
        left_list = sorted(left_dir.glob("*.png"))
        right_list = sorted(right_dir.glob("*.png"))
        if not left_list:
            raise SystemExit("Nenhuma imagem em images/stereoLeft/")
        left, right = left_list[0], right_list[0]

    img_l = cv2.imread(str(left))
    img_r = cv2.imread(str(right))
    if img_l is None or img_r is None:
        raise SystemExit(f"Falha ao ler {left} ou {right}")

    rect_l, rect_r = calibration.undistort_rectify(img_l, img_r)

    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    black_l = (gray_l < 5).mean() * 100
    black_r = (gray_r < 5).mean() * 100
    corr = row_correlation(rect_l, rect_r)

    print("\n--- retificação ---")
    print(f"  Par: {left.name} + {right.name}")
    print(f"  Área preta: esq {black_l:.1f}% | dir {black_r:.1f}%  (ideal < 30%)")
    print(f"  Correlação epipolar: {corr:.3f}  (ideal > 0.7)")

    scale = 0.75
    small_l = cv2.resize(rect_l, None, fx=scale, fy=scale)
    small_r = cv2.resize(rect_r, None, fx=scale, fy=scale)
    pad_h = max(small_l.shape[0], small_r.shape[0])

    def pad(img: np.ndarray) -> np.ndarray:
        out = np.zeros((pad_h, img.shape[1], 3), dtype=np.uint8)
        out[: img.shape[0], :] = img
        return out

    combined = np.hstack([pad(small_l), pad(small_r)])
    cv2.putText(combined, "ESQ retificada | DIR retificada", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    label = f"corr={corr:.2f}  preto L={black_l:.0f}% R={black_r:.0f}%"
    cv2.putText(combined, label, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    out_path = PROJECT_ROOT / "calibration_check.png"
    cv2.imwrite(str(out_path), combined)
    print(f"\n  Imagem salva: {out_path}")

    ok = (
        corr == corr
        and corr > 0.5
        and black_l < 40
        and black_r < 40
    )
    if ok:
        print("\nResultado: CALIBRAÇÃO PARECE BOA — reinicie: python src/worker.py serve")
    else:
        print("\nResultado: CALIBRAÇÃO RUIM — capture mais fotos e rode stereo_calibration.py")


if __name__ == "__main__":
    main()
