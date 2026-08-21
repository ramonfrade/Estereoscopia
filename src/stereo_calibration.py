"""Calibração estéreo: modo ao vivo (padrão) ou a partir de imagens salvas.

Modo ao vivo — detecção autônoma do tabuleiro (8x8 ou 8x4), sem tirar fotos
antes; dimensões do tabuleiro e baseline informadas por parâmetro:

  python src/stereo_calibration.py --board-width 80 --board-height 40 --baseline 7.0

  Aponta o tabuleiro para as câmeras: o padrão é detectado sozinho e o erro
  percentual (2 casas) aparece ao vivo. Tecla 's' captura o par; ESC/q
  finaliza e calibra.

Fallback — a partir de imagens salvas (calibration_images.py):

  python src/stereo_calibration.py --from-images [--size 7 7]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chessboard_auto import board_error_pct, detect_chessboard_auto, square_size_cm

PROJECT_ROOT = SRC.parent
SHOW_PREVIEW = bool(os.environ.get("DISPLAY"))

FRAME_SIZE = (640, 480)
LEGACY_SQUARE_CM = 4.4  # usado apenas no fallback sem --board-width/--board-height
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def build_objp(chessboard_size: tuple[int, int], square_cm: float) -> np.ndarray:
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
    objp[:, :2] *= square_cm
    return objp


def run_calibration(
    objpoints: list,
    imgpoints_l: list,
    imgpoints_r: list,
    frame_size: tuple[int, int],
    chessboard_size: tuple[int, int],
    square_cm: float,
    meta: dict,
    calibration_error_pct: float,
    capture_errors: list[dict],
) -> float:
    """Calibração mono + estéreo, retificação e gravação de stereoMap.xml e calibration.json."""
    height, width = frame_size[1], frame_size[0]

    retL, camera_matrix_l, dist_l, _, _ = cv.calibrateCamera(
        objpoints, imgpoints_l, frame_size, None, None
    )
    new_camera_l, _ = cv.getOptimalNewCameraMatrix(
        camera_matrix_l, dist_l, (width, height), 1, (width, height)
    )

    retR, camera_matrix_r, dist_r, _, _ = cv.calibrateCamera(
        objpoints, imgpoints_r, frame_size, None, None
    )
    new_camera_r, _ = cv.getOptimalNewCameraMatrix(
        camera_matrix_r, dist_r, (width, height), 1, (width, height)
    )

    flags = cv.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret_stereo, new_camera_l, dist_l, new_camera_r, dist_r, rot, trans, _, _ = (
        cv.stereoCalibrate(
            objpoints,
            imgpoints_l,
            imgpoints_r,
            new_camera_l,
            dist_l,
            new_camera_r,
            dist_r,
            (width, height),
            criteria_stereo,
            flags,
        )
    )

    rectify_scale = 1
    rect_l, rect_r, proj_l, proj_r, q, _, _ = cv.stereoRectify(
        new_camera_l, dist_l, new_camera_r, dist_r, (width, height), rot, trans, rectify_scale
    )

    stereo_map_l = cv.initUndistortRectifyMap(
        new_camera_l, dist_l, rect_l, proj_l, (width, height), cv.CV_16SC2
    )
    stereo_map_r = cv.initUndistortRectifyMap(
        new_camera_r, dist_r, rect_r, proj_r, (width, height), cv.CV_16SC2
    )

    print("Salvando stereoMap.xml ...")
    stereo_map_path = PROJECT_ROOT / "stereoMap.xml"
    cv_file = cv.FileStorage(str(stereo_map_path), cv.FILE_STORAGE_WRITE)
    cv_file.write("stereoMapL_x", stereo_map_l[0])
    cv_file.write("stereoMapL_y", stereo_map_l[1])
    cv_file.write("stereoMapR_x", stereo_map_r[0])
    cv_file.write("stereoMapR_y", stereo_map_r[1])
    cv_file.release()

    calibration_data = {
        "Q": q.tolist(),
        "frameSize": [width, height],
        "baseline_units": float(np.linalg.norm(trans)),
        "physical_baseline_cm": float(meta["baseline_cm"]),
        "chessboardSize": list(chessboard_size),
        "pattern": meta["pattern"],
        "board_size_cm": meta["board_size_cm"],
        "square_cm": round(square_cm, 3),
        "cameras": meta["cameras"],
        "valid_pairs": len(objpoints),
        "stereo_rms_px": round(float(ret_stereo), 3),
        "capture_errors": capture_errors,
        "calibration_error_pct": round(calibration_error_pct, 2),
    }
    calib_path = PROJECT_ROOT / "calibration.json"
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2)

    print(f"Calibração salva em {stereo_map_path} e {calib_path}")
    print(
        f"Pares válidos: {len(objpoints)} | RMS estéreo: {ret_stereo:.3f} px | "
        f"erro de calibração: {calibration_error_pct:.2f}%"
    )
    return float(ret_stereo)


def _open_cameras(left: int, right: int) -> dict[int, cv.VideoCapture]:
    caps: dict[int, cv.VideoCapture] = {}
    for index in (left, right):
        cap = cv.VideoCapture(index)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera /dev/video{index}")
        caps[index] = cap
    time.sleep(1.5)
    return caps


def calibrate_live(args) -> None:
    if not SHOW_PREVIEW:
        sys.exit("Modo ao vivo requer DISPLAY (janelas OpenCV).")
    if args.board_width is None or args.board_height is None or args.baseline is None:
        sys.exit("Modo ao vivo exige --board-width, --board-height e --baseline (em cm).")

    caps = _open_cameras(args.left, args.right)
    print(
        f"Tabuleiro 8x8 ou 8x4 (detecção autônoma). "
        f"'s' captura par | ESC/q finaliza e calibra | baseline {args.baseline} cm"
    )

    pattern: tuple[int, int] | None = None
    objpoints: list = []
    imgpoints_l: list = []
    imgpoints_r: list = []
    capture_errors: list[dict] = []
    current: dict | None = None

    try:
        while True:
            for index in (args.left, args.right):
                caps[index].grab()
            frames: dict[int, np.ndarray] = {}
            for index in (args.left, args.right):
                ok, frame = caps[index].retrieve()
                if not ok:
                    raise RuntimeError(f"Falha na leitura da câmera /dev/video{index}")
                frames[index] = frame
            f_l, f_r = frames[args.left], frames[args.right]
            g_l = cv.cvtColor(f_l, cv.COLOR_BGR2GRAY)
            g_r = cv.cvtColor(f_r, cv.COLOR_BGR2GRAY)

            res_l = detect_chessboard_auto(g_l)
            res_r = detect_chessboard_auto(g_r)
            current = None

            if res_l is None or res_r is None:
                msg = "Tabuleiro nao detectado em ambas as cameras"
                cv.putText(f_l, msg, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.putText(f_r, msg, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                size_l, corners_l = res_l
                size_r, corners_r = res_r
                if size_l != size_r:
                    msg = f"Padroes diferentes: L {size_l} vs R {size_r}"
                    cv.putText(f_l, msg, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv.putText(f_r, msg, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    err_l = board_error_pct(g_l, corners_l, size_l)
                    err_r = board_error_pct(g_r, corners_r, size_r)
                    cols, rows = size_l
                    label = (
                        f"Tabuleiro {cols + 1}x{rows + 1} | erro L: {err_l:.2f}% "
                        f"R: {err_r:.2f}% | capturados: {len(objpoints)}"
                    )
                    cv.putText(f_l, label, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv.putText(f_r, label, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if pattern is None:
                        pattern = size_l
                    current = {
                        "size": size_l,
                        "corners_l": corners_l,
                        "corners_r": corners_r,
                        "err_l": err_l,
                        "err_r": err_r,
                    }

            cv.imshow("Calibracao esq", f_l)
            cv.imshow("Calibracao dir", f_r)
            key = cv.waitKey(1) & 0xFF

            if key in (ord("s"), ord("S")) and current is not None:
                if current["size"] == pattern:
                    square = square_size_cm(args.board_width, args.board_height, pattern)
                    objpoints.append(build_objp(pattern, square))
                    imgpoints_l.append(current["corners_l"])
                    imgpoints_r.append(current["corners_r"])
                    capture_errors.append(
                        {
                            "left": round(current["err_l"], 2),
                            "right": round(current["err_r"], 2),
                        }
                    )
                    print(
                        f"Par {len(objpoints)} capturado "
                        f"(erro L {current['err_l']:.2f}% R {current['err_r']:.2f}%)"
                    )
                else:
                    cols, rows = current["size"]
                    print(
                        f"AVISO: padrão {cols + 1}x{rows + 1} difere do primeiro "
                        f"({pattern}) — par ignorado"
                    )

            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        for cap in caps.values():
            cap.release()
        cv.destroyAllWindows()

    if not objpoints:
        sys.exit("Nenhum par capturado — calibração abortada.")

    assert pattern is not None
    square = square_size_cm(args.board_width, args.board_height, pattern)
    calib_error = float(np.mean([(e["left"] + e["right"]) / 2 for e in capture_errors]))
    meta = {
        "baseline_cm": args.baseline,
        "pattern": f"{pattern[0] + 1}x{pattern[1] + 1}",
        "board_size_cm": [args.board_width, args.board_height],
        "cameras": {
            "calibLeft": args.left,
            "calibRight": args.right,
            "physicalLeft": args.left,
            "physicalRight": args.right,
        },
    }
    run_calibration(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        FRAME_SIZE,
        pattern,
        square,
        meta,
        calib_error,
        capture_errors,
    )


def calibrate_from_images(args) -> None:
    images_left = sorted(glob.glob(str(PROJECT_ROOT / "images/stereoLeft/*.png")))
    images_right = sorted(glob.glob(str(PROJECT_ROOT / "images/stereoRight/*.png")))
    if not images_left:
        sys.exit("Nenhuma imagem em images/stereoLeft/ — capture antes com calibration_images.py")

    size = (args.size[0], args.size[1])
    if args.board_width is not None and args.board_height is not None:
        square = square_size_cm(args.board_width, args.board_height, size)
        print(f"Quadrado: {square:.2f} cm (de --board-width/--board-height)")
    else:
        square = LEGACY_SQUARE_CM
        print(f"AVISO: sem --board-width/--board-height — usando quadrado fixo {square} cm")

    objpoints: list = []
    imgpoints_l: list = []
    imgpoints_r: list = []

    for img_left, img_right in zip(images_left, images_right):
        img_l = cv.imread(img_left)
        img_r = cv.imread(img_right)
        gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

        ret_l, corners_l = cv.findChessboardCorners(gray_l, size, None)
        ret_r, corners_r = cv.findChessboardCorners(gray_r, size, None)
        if ret_l and ret_r:
            objpoints.append(build_objp(size, square))
            corners_l = cv.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), CRITERIA)
            imgpoints_l.append(corners_l)
            corners_r = cv.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), CRITERIA)
            imgpoints_r.append(corners_r)
            if SHOW_PREVIEW:
                cv.drawChessboardCorners(img_l, size, corners_l, ret_l)
                cv.imshow("Img L", img_l)
                cv.drawChessboardCorners(img_r, size, corners_r, ret_r)
                cv.imshow("Img R", img_r)
                cv.waitKey(300)
        else:
            print(f"Ignorado (padrão não encontrado): {Path(img_left).name} + {Path(img_right).name}")

    if SHOW_PREVIEW:
        cv.destroyAllWindows()

    print(f"Pares válidos: {len(objpoints)}")
    if not objpoints:
        raise SystemExit("Nenhum canto detectado. Verifique --size e as imagens.")

    meta = {
        "baseline_cm": args.baseline,
        "pattern": f"{size[0] + 1}x{size[1] + 1}",
        "board_size_cm": (
            [args.board_width, args.board_height]
            if args.board_width is not None
            else [None, None]
        ),
        "cameras": {
            "calibLeft": args.left,
            "calibRight": args.right,
            "physicalLeft": args.right,
            "physicalRight": args.left,
        },
    }
    run_calibration(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        FRAME_SIZE,
        size,
        square,
        meta,
        calibration_error_pct=0.0,
        capture_errors=[],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibração estéreo (ao vivo por padrão)")
    parser.add_argument("--from-images", action="store_true", help="usar imagens salvas (fallback)")
    parser.add_argument(
        "--size", type=int, nargs=2, default=[7, 7], metavar=("COLS", "ROWS"),
        help="cantos internos para --from-images (padrão 7 7)",
    )
    parser.add_argument("--board-width", type=float, help="largura do tabuleiro em cm")
    parser.add_argument("--board-height", type=float, help="altura do tabuleiro em cm")
    parser.add_argument("--baseline", type=float, help="distância entre focos das câmeras em cm")
    parser.add_argument("--left", type=int, default=0, help="índice da câmera esquerda (padrão 0)")
    parser.add_argument("--right", type=int, default=2, help="índice da câmera direita (padrão 2)")
    args = parser.parse_args()

    if args.from_images:
        calibrate_from_images(args)
    else:
        calibrate_live(args)


if __name__ == "__main__":
    main()
