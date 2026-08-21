"""Motor estéreo: captura, retificação e profundidade (face ou SGBM)."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np

import calibration
from depth_config import CALIBRATION_JSON, DepthConfig, DepthMode
from depth_strategies import FaceDepthStrategy, SgbmDepthStrategy

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CALIB_LEFT = 0
DEFAULT_CALIB_RIGHT = 2
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PHYSICAL_BASELINE_CM = 15.0


class CameraLayout:
    """Separa ordem da calibração (fixa) da posição física na bancada."""

    def __init__(
        self,
        calib_left: int = DEFAULT_CALIB_LEFT,
        calib_right: int = DEFAULT_CALIB_RIGHT,
        physical_left: int | None = None,
        physical_right: int | None = None,
    ):
        self.calib_left = calib_left
        self.calib_right = calib_right
        self.physical_left = physical_left if physical_left is not None else calib_left
        self.physical_right = physical_right if physical_right is not None else calib_right

    @classmethod
    def from_calibration_json(cls, data: dict) -> CameraLayout:
        cameras = data.get("cameras", {})
        return cls(
            calib_left=int(cameras.get("calibLeft", DEFAULT_CALIB_LEFT)),
            calib_right=int(cameras.get("calibRight", DEFAULT_CALIB_RIGHT)),
            physical_left=int(cameras.get("physicalLeft", cameras.get("calibLeft", DEFAULT_CALIB_LEFT))),
            physical_right=int(cameras.get("physicalRight", cameras.get("calibRight", DEFAULT_CALIB_RIGHT))),
        )

    @property
    def indices(self) -> tuple[int, int]:
        return self.calib_left, self.calib_right


class StereoEngine:
    def __init__(
        self,
        layout: CameraLayout | None = None,
        depth_mode: DepthMode = DepthMode.FACE,
        depth_config: DepthConfig | None = None,
    ):
        self.layout = layout
        self.depth_mode = depth_mode
        self.depth_config = depth_config or DepthConfig.from_calibration_json()
        self.caps: dict[int, cv2.VideoCapture] = {}
        self.q_matrix: np.ndarray | None = None
        self.frame_size: tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT)
        self.unit_to_cm = 1.0
        self._stereo = self._create_stereo_matcher()
        self._face_strategy: FaceDepthStrategy | None = None
        self._sgbm_strategy: SgbmDepthStrategy | None = None
        self._load_calibration()

    def _load_calibration(self) -> None:
        if not CALIBRATION_JSON.exists():
            raise FileNotFoundError(
                f"{CALIBRATION_JSON} não encontrado. Execute: python src/stereo_calibration.py"
            )
        data = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
        if self.layout is None:
            self.layout = CameraLayout.from_calibration_json(data)

        self.q_matrix = np.array(data["Q"], dtype=np.float64)
        w, h = data["frameSize"]
        self.frame_size = (int(w), int(h))
        baseline_units = float(data.get("baseline_units", data.get("baseline_m", 1.0)))
        physical_baseline = float(data.get("physical_baseline_cm", PHYSICAL_BASELINE_CM))
        if baseline_units > 0:
            self.unit_to_cm = physical_baseline / baseline_units

        if self.depth_mode == DepthMode.FACE:
            self._face_strategy = FaceDepthStrategy(self.depth_config)
        else:
            assert self.q_matrix is not None
            self._sgbm_strategy = SgbmDepthStrategy(self._stereo, self.q_matrix, self.unit_to_cm)

    @staticmethod
    def _create_stereo_matcher() -> cv2.StereoSGBM:
        window_size = 5
        min_disp = 0
        num_disp = 16 * 8
        return cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=50,
            speckleRange=16,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def open_cameras(self) -> None:
        assert self.layout is not None
        needed = {self.layout.calib_left, self.layout.calib_right}
        for index in needed:
            cap = cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
            if not cap.isOpened():
                raise RuntimeError(f"Não foi possível abrir a câmera /dev/video{index}")
            self.caps[index] = cap
        time.sleep(1.5)

    def close_cameras(self) -> None:
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()
        if self._face_strategy is not None:
            self._face_strategy.close()

    def capture_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """Captura na ordem da calibração: (video calibLeft, video calibRight)."""
        if not self.caps:
            raise RuntimeError("Câmeras não abertas. Chame open_cameras() primeiro.")
        assert self.layout is not None

        indices = (self.layout.calib_left, self.layout.calib_right)
        for _ in range(2):
            for index in indices:
                self.caps[index].grab()

        frames: dict[int, np.ndarray] = {}
        for index in indices:
            ok, frame = self.caps[index].retrieve()
            if not ok:
                raise RuntimeError(f"Falha na leitura da câmera /dev/video{index}")
            frames[index] = frame

        return frames[self.layout.calib_left], frames[self.layout.calib_right]

    @staticmethod
    def find_closest_point(
        disparity: np.ndarray, mask: np.ndarray | None = None, margin: float = 0.08
    ) -> tuple[int, int]:
        return SgbmDepthStrategy.find_closest_point(disparity, mask, margin)

    @staticmethod
    def _rectified_for_physical(
        rect_calib_left: np.ndarray,
        rect_calib_right: np.ndarray,
        layout: CameraLayout,
    ) -> tuple[np.ndarray, np.ndarray]:
        rect_by_index = {
            layout.calib_left: rect_calib_left,
            layout.calib_right: rect_calib_right,
        }
        return rect_by_index[layout.physical_left], rect_by_index[layout.physical_right]

    def process_pair(
        self,
        frame_calib_left: np.ndarray,
        frame_calib_right: np.ndarray,
        sample_x: int | None = None,
        sample_y: int | None = None,
    ) -> dict:
        assert self.layout is not None

        rect_calib_left, rect_calib_right = calibration.undistort_rectify(
            frame_calib_left, frame_calib_right
        )
        display_left, display_right = self._rectified_for_physical(
            rect_calib_left, rect_calib_right, self.layout
        )

        if self.depth_mode == DepthMode.FACE:
            assert self._face_strategy is not None
            depth_result = self._face_strategy.estimate(
                display_left, display_right, rect_calib_left, rect_calib_right
            )
        else:
            assert self._sgbm_strategy is not None
            depth_result = self._sgbm_strategy.estimate(
                display_left,
                display_right,
                rect_calib_left,
                rect_calib_right,
                sample_x,
                sample_y,
            )

        return {
            "rect_calib_left": rect_calib_left,
            "rect_calib_right": rect_calib_right,
            "display_left": depth_result["display_left"],
            "display_right": depth_result["display_right"],
            "disparity": depth_result["disparity"],
            "depth_cm": depth_result["depth_cm"],
            "disparity_px": depth_result["disparity_px"],
            "sample_point": depth_result["sample_point"],
            "track_mode": depth_result["track_mode"],
            "depth_mode": self.depth_mode.value,
        }

    def process_frame(self, sample_x: int | None = None, sample_y: int | None = None) -> dict:
        frame_calib_left, frame_calib_right = self.capture_pair()
        result = self.process_pair(frame_calib_left, frame_calib_right, sample_x, sample_y)
        result["frame_calib_left"] = frame_calib_left
        result["frame_calib_right"] = frame_calib_right
        return result


def encode_image_b64(image: np.ndarray, fmt: str = ".jpg") -> str:
    ok, buffer = cv2.imencode(fmt, image)
    if not ok:
        raise RuntimeError("Falha ao codificar imagem.")
    return base64.b64encode(buffer).decode("ascii")


def disparity_to_colormap(disparity: np.ndarray) -> np.ndarray:
    if not np.any(disparity > 0):
        return np.zeros((*disparity.shape, 3), dtype=np.uint8)
    normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)


def annotate_frame(
    frame: np.ndarray, depth_cm: float | None, point: dict, track_mode: str = ""
) -> np.ndarray:
    output = frame.copy()
    x, y = int(point["x"]), int(point["y"])
    cv2.drawMarker(output, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(output, (x, y), 10, (0, 255, 0), 2)
    if depth_cm is not None and depth_cm != float("inf"):
        label = f"Distancia: {depth_cm:.1f} cm"
    else:
        label = "Distancia: --"
    cv2.putText(output, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    mode_labels = {
        "face": "Rosto detectado",
        "lost": "TRACKING LOST",
        "closest": "Objeto mais proximo",
        "motion": "Seguindo movimento",
        "manual": "Ponto clicado",
    }
    sub = mode_labels.get(track_mode, "")
    if sub:
        color = (0, 0, 255) if track_mode == "lost" else (0, 255, 255)
        cv2.putText(output, sub, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output
