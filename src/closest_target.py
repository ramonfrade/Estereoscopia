"""
Alvo mais próximo + rostos — janela única (câmera esquerda física).

Usa as DUAS câmeras (par retificado + disparidade densa SGBM) para calcular
distâncias, mas exibe apenas o feed esquerdo com:
  - círculo pequeno verde no ALVO MAIS PRÓXIMO da câmera (dedo, mão ou
    qualquer objeto — não é preciso reconhecer o objeto) + distância em cm;
  - retângulo verde em TODOS os rostos detectados (MediaPipe).

Uso:
  python src/closest_target.py
  python src/closest_target.py --margin 0.08 --min-area-pct 0.002

Teclas: 'q' sai.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from depth_config import DepthMode
from face_detection import FaceDetector
from stereo_engine import StereoEngine

MIN_VALID_DISPARITY = 0.5
MAX_DEPTH_CM = 1_000.0


def find_closest_target(
    disparity: np.ndarray,
    margin: float = 0.08,
    min_area_pct: float = 0.002,
    near_percentile: float = 70.0,
    max_disparity: float | None = None,
    min_x: int = 0,
) -> tuple[int, int] | None:
    """Centróide do componente conectado mais próximo (maior disparidade média).

    - Descarta a borda (margin) onde a disparidade é ruidosa;
    - Mantém apenas disparidades válidas (> 0.5 px);
    - Ignora saturação da faixa SGBM (`max_disparity`) e a banda esquerda
      inválida (`min_x`, colunas x < numDisparities);
    - Considera "perto" valores acima do percentil `near_percentile`;
    - Escolhe o componente com maior disparidade média e área >= min_area_pct;
    - Retorna (x, y) em coordenadas do frame original, ou None sem alvo.
    """
    h, w = disparity.shape
    y1, y2 = int(h * margin), int(h * (1 - margin))
    x1, x2 = int(w * margin), int(w * (1 - margin))
    roi = disparity[y1:y2, x1:x2]

    valid = roi > MIN_VALID_DISPARITY
    if max_disparity is not None:
        valid &= roi < max_disparity
    if min_x > x1:
        valid &= np.arange(x1, x2)[np.newaxis, :] >= min_x
    if not np.any(valid):
        return None

    threshold = float(np.percentile(roi[valid], near_percentile))
    near = (valid & (roi >= threshold)).astype(np.uint8)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(near, connectivity=8)
    min_area = int(min_area_pct * roi.size)

    best_comp = -1
    best_mean = -1.0
    for comp in range(1, num):
        area = int(stats[comp, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mean_d = float(np.mean(roi[labels == comp]))
        if mean_d > best_mean:
            best_mean = mean_d
            best_comp = comp

    if best_comp < 0:
        return None
    cx = int(round(centroids[best_comp][0])) + x1
    cy = int(round(centroids[best_comp][1])) + y1
    return cx, cy


def depth_at_point(
    disparity: np.ndarray,
    q_matrix: np.ndarray,
    unit_to_cm: float,
    x: int,
    y: int,
    window: int = 15,
    max_disparity: float | None = None,
) -> float | None:
    """Profundidade em cm no ponto (x, y) via matriz Q (mesma matemática da engine)."""
    half = window // 2
    y1, y2 = max(0, y - half), min(disparity.shape[0], y + half + 1)
    x1, x2 = max(0, x - half), min(disparity.shape[1], x + half + 1)
    patch = disparity[y1:y2, x1:x2]
    valid = patch[patch > MIN_VALID_DISPARITY]
    if max_disparity is not None:
        valid = valid[valid < max_disparity]
    if valid.size == 0:
        return None

    d = float(np.median(valid))
    q = q_matrix
    w = q[3, 2] * d + q[3, 3]
    if abs(w) < 1e-6:
        return None
    z_units = q[2, 3] / w
    depth_cm = abs(z_units) * unit_to_cm
    if depth_cm <= 0 or depth_cm > MAX_DEPTH_CM:
        return None
    return depth_cm


def run_demo(margin: float = 0.08, min_area_pct: float = 0.002, num_disparities: int = 512) -> None:
    eng = StereoEngine(depth_mode=DepthMode.SGBM, num_disparities=num_disparities)
    eng.open_cameras()
    detector = FaceDetector()
    print(
        f"Alvo mais próximo (janela única — câmera esquerda). "
        f"Faixa SGBM {eng.num_disparities} px (centro válido). 'q' para sair."
    )
    try:
        while True:
            start = time.time()
            result = eng.process_frame()
            disparity = result["disparity"]
            left = result["display_left"]

            detector.detect_all_with_draw(left)  # retângulos em todos os rostos

            nd = eng.num_disparities
            max_d = float(nd - 1)
            target = find_closest_target(
                disparity, margin=margin, min_area_pct=min_area_pct,
                max_disparity=max_d, min_x=nd,
            )
            depth_cm: float | None = None
            if target is not None:
                cx, cy = target
                depth_cm = depth_at_point(
                    disparity, eng.q_matrix, eng.unit_to_cm, cx, cy, max_disparity=max_d
                )
                cv2.circle(left, (cx, cy), 8, (0, 255, 0), 2)
                if depth_cm is not None:
                    label = f"{depth_cm:.1f} cm"
                else:
                    label = "muito proximo"
                cv2.putText(left, label, (cx + 14, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(left, "Sem alvo proximo", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            elapsed = time.time() - start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            cv2.putText(left, f"FPS {fps:.1f}", (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Alvo mais proximo (esquerda)", left)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            print(f"FPS: {fps:.1f} | Alvo: {target} | Distância: "
                  f"{depth_cm if depth_cm is None else f'{depth_cm:.1f} cm'}")
    finally:
        eng.close_cameras()
        detector.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alvo mais próximo + rostos em janela única (câmera esquerda)"
    )
    parser.add_argument("--margin", type=float, default=0.08,
                        help="fração da borda ignorada na busca (padrão 0.08)")
    parser.add_argument("--min-area-pct", type=float, default=0.002,
                        help="área mínima do alvo como fração do frame (padrão 0.002)")
    parser.add_argument("--num-disparities", type=int, default=512,
                        help="faixa de disparidade pedida (múltiplo de 16). Limitada a "
                             "metade da largura do frame para o centro continuar "
                             "mensurável; padrão 512 → 320 em 640 px")
    args = parser.parse_args()
    run_demo(margin=args.margin, min_area_pct=args.min_area_pct,
             num_disparities=args.num_disparities)


if __name__ == "__main__":
    main()
