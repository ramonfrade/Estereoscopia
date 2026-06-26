"""
Testa detecção do tabuleiro nas câmeras ou em uma foto.

Uso:
  python src/detect_chessboard.py
  python src/detect_chessboard.py --image images/stereoLeft/imageL0.png
  python src/detect_chessboard.py --size 7 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

SRC = Path(__file__).resolve().parent
PROJECT_ROOT = SRC.parent

FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--size", type=int, nargs=2, default=[7, 7], metavar=("COLS", "ROWS"))
    args = parser.parse_args()
    size = (args.size[0], args.size[1])
    if args.image:
        test_image(args.image, size)
    else:
        test_live(size)


if __name__ == "__main__":
    main()
