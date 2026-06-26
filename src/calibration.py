"""Carrega mapas de retificação estéreo gerados por stereo_calibration.py."""

from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEREO_MAP_PATH = PROJECT_ROOT / "stereoMap.xml"

_maps = None


def _load_maps():
    global _maps
    if _maps is not None:
        return _maps

    if not STEREO_MAP_PATH.exists():
        raise FileNotFoundError(
            f"stereoMap.xml não encontrado em {STEREO_MAP_PATH}. "
            "Execute: python src/stereo_calibration.py"
        )

    fs = cv2.FileStorage(str(STEREO_MAP_PATH), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Não foi possível abrir {STEREO_MAP_PATH}")

    _maps = {
        "L_x": fs.getNode("stereoMapL_x").mat(),
        "L_y": fs.getNode("stereoMapL_y").mat(),
        "R_x": fs.getNode("stereoMapR_x").mat(),
        "R_y": fs.getNode("stereoMapR_y").mat(),
    }
    fs.release()
    return _maps


def rectify_calib_left(frame):
    """Retifica frame da câmera calibrada como esquerda (video0 / imageL)."""
    maps = _load_maps()
    return cv2.remap(
        frame, maps["L_x"], maps["L_y"], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0
    )


def rectify_calib_right(frame):
    """Retifica frame da câmera calibrada como direita (video2 / imageR)."""
    maps = _load_maps()
    return cv2.remap(
        frame, maps["R_x"], maps["R_y"], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0
    )


def undistort_rectify(frame_calib_left, frame_calib_right):
    """Retifica par na ordem da calibração (esq=video0, dir=video2)."""
    return rectify_calib_left(frame_calib_left), rectify_calib_right(frame_calib_right)
