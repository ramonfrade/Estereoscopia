"""
Mede o tamanho de objetos sobre fundo homogêneo (OpenCV).

Uso:
  python src/measure_object_size.py --image foto.jpg --pixel-cm-ratio 10.5

O `--pixel-cm-ratio` (pixels por centímetro) é obtido uma vez por montagem:
fotografe um objeto de referência de tamanho conhecido (ex.: moeda de 2,5 cm),
meça sua largura em pixels e divida pelos cm (ex.: 26 px / 2,5 cm = 10,4).

Desenha as caixas com as dimensões, salva em `--output` e imprime os valores.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

from object_detector import HomogeneousBgDetector, object_size_cm

SHOW_PREVIEW = bool(os.environ.get("DISPLAY"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mede tamanho de objetos em fundo homogêneo")
    parser.add_argument("--image", type=Path, required=True, help="imagem de entrada")
    parser.add_argument(
        "--pixel-cm-ratio",
        type=float,
        required=True,
        help="pixels por centímetro (medidos com objeto de referência)",
    )
    parser.add_argument("--min-area", type=float, default=2000, help="área mínima do contorno")
    parser.add_argument("--output", type=Path, default=None, help="imagem de saída (padrão: <imagem>_measured.png)")
    args = parser.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"Falha ao ler a imagem: {args.image}")

    detector = HomogeneousBgDetector()
    contours = detector.detect_objects(img, min_area=args.min_area)
    if not contours:
        print("Nenhum objeto detectado (fundo não homogêneo? ajuste --min-area).")
        return

    for cnt in contours:
        w_cm, h_cm = object_size_cm(cnt, args.pixel_cm_ratio)
        rect = cv2.minAreaRect(cnt)
        (x, y), (w_px, h_px), angle = rect
        print(f"Largura: {w_cm:.2f} cm | Altura: {h_cm:.2f} cm")

        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        cv2.putText(
            img,
            f"Largura {w_cm:.2f} cm",
            (int(x - w_px / 2), int(y - 20)),
            cv2.FONT_HERSHEY_PLAIN,
            1.2,
            (0, 200, 0),
            2,
        )
        cv2.putText(
            img,
            f"Altura {h_cm:.2f} cm",
            (int(x - w_px / 2), int(y)),
            cv2.FONT_HERSHEY_PLAIN,
            1.2,
            (0, 200, 0),
            2,
        )

    output = args.output or args.image.with_name(f"{args.image.stem}_measured.png")
    cv2.imwrite(str(output), img)
    print(f"Imagem salva em {output}")

    if SHOW_PREVIEW:
        cv2.imshow("Medicao", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
