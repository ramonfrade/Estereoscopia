
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np

import calibration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_JSON = PROJECT_ROOT / "calibration.json"

# Padrão: mesmos índices usados em calibration_images.py / stereo_calibration.py
DEFAULT_CALIB_LEFT = 0
DEFAULT_CALIB_RIGHT = 2
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PHYSICAL_BASELINE_CM = 9.0


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
    def __init__(self, layout: CameraLayout | None = None):
        self.layout = layout
        self.caps: dict[int, cv2.VideoCapture] = {}
        self.q_matrix: np.ndarray | None = None
        self.frame_size: tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT)
        self.unit_to_cm = 1.0
        self._stereo = self._create_stereo_matcher()
        self._prev_gray: np.ndarray | None = None
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

    def capture_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """Captura na ordem da calibração: (video calibLeft, video calibRight)."""
        if not self.caps:
            raise RuntimeError("Câmeras não abertas. Chame open_cameras() primeiro.")
        assert self.layout is not None

        indices = (self.layout.calib_left, self.layout.calib_right)
        # Descarta buffers antigos para frames mais recentes e melhor sincronia USB
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
    def find_closest_point(disparity: np.ndarray, mask: np.ndarray | None = None, margin: float = 0.08) -> tuple[int, int]:
        """
        Ponto com maior disparidade válida. Máscara opcional (ex.: região em movimento).
        """
        h, w = disparity.shape
        y1, y2 = int(h * margin), int(h * (1 - margin))
        x1, x2 = int(w * margin), int(w * (1 - margin))
        roi = disparity[y1:y2, x1:x2].copy()
        roi[roi <= 0.5] = 0

        if mask is not None:
            roi_mask = mask[y1:y2, x1:x2]
            roi[~roi_mask] = 0

        if not np.any(roi > 0):
            return w // 2, h // 2
        flat_idx = int(np.argmax(roi))
        ly, lx = np.unravel_index(flat_idx, roi.shape)
        return x1 + lx, y1 + ly

    def _motion_mask(self, gray_left: np.ndarray) -> np.ndarray | None:
        if self._prev_gray is None:
            self._prev_gray = gray_left.copy()
            return None

        diff = cv2.absdiff(gray_left, self._prev_gray)
        _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion = cv2.dilate(motion, np.ones((7, 7), np.uint8), iterations=2)
        self._prev_gray = gray_left.copy()
        return motion > 0

    def depth_at_point(
        self, disparity: np.ndarray, x: int, y: int, window: int = 15
    ) -> tuple[float | None, float]:
        """Profundidade (cm) e disparidade mediana numa janela ao redor do ponto."""
        if self.q_matrix is None:
            return None, 0.0

        half = window // 2
        y1, y2 = max(0, y - half), min(disparity.shape[0], y + half + 1)
        x1, x2 = max(0, x - half), min(disparity.shape[1], x + half + 1)
        patch = disparity[y1:y2, x1:x2]
        valid = patch[patch > 0.5]
        if valid.size == 0:
            return None, 0.0

        d = float(np.median(valid))
        q = self.q_matrix
        w = q[3, 2] * d + q[3, 3]
        if abs(w) < 1e-6:
            return None, d
        z_units = q[2, 3] / w
        depth_cm = abs(z_units) * self.unit_to_cm
        if depth_cm <= 0 or depth_cm > 1_000:
            return None, d
        return depth_cm, d

    @staticmethod
    def _rectified_for_physical(
        rect_calib_left: np.ndarray,
        rect_calib_right: np.ndarray,
        layout: CameraLayout,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mapeia imagens retificadas para esquerda/direita físicas na bancada."""
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

        gray_left = cv2.cvtColor(rect_calib_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rect_calib_right, cv2.COLOR_BGR2GRAY)

        # Disparidade sempre na ordem da calibração (esq → dir)
        disparity_raw = self._stereo.compute(gray_left, gray_right)
        disparity = disparity_raw.astype(np.float32) / 16.0

        h, w = gray_left.shape

        if sample_x is not None and sample_y is not None:
            cx = max(0, min(w - 1, int(sample_x)))
            cy = max(0, min(h - 1, int(sample_y)))
            track_mode = "manual"
        else:
            motion_mask = self._motion_mask(gray_left)
            if motion_mask is not None and np.any(motion_mask):
                cx, cy = self.find_closest_point(disparity, mask=motion_mask)
                track_mode = "motion"
            else:
                cx, cy = self.find_closest_point(disparity)
                track_mode = "closest"

        depth_cm, disp_value = self.depth_at_point(disparity, cx, cy)

        return {
            "rect_calib_left": rect_calib_left,
            "rect_calib_right": rect_calib_right,
            "display_left": display_left,
            "display_right": display_right,
            "disparity": disparity,
            "depth_cm": depth_cm,
            "disparity_px": disp_value,
            "sample_point": {"x": int(cx), "y": int(cy)},
            "track_mode": track_mode,
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
    normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)


def annotate_frame(frame: np.ndarray, depth_cm: float | None, point: dict, track_mode: str = "") -> np.ndarray:
    output = frame.copy()
    x, y = int(point["x"]), int(point["y"])
    cv2.drawMarker(output, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(output, (x, y), 10, (0, 255, 0), 2)
    if depth_cm is not None:
        label = f"Distancia: {depth_cm:.1f} cm"
    else:
        label = "Distancia: --"
    cv2.putText(output, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    if track_mode == "closest":
        cv2.putText(output, "Objeto mais proximo", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif track_mode == "motion":
        cv2.putText(output, "Seguindo movimento", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif track_mode == "manual":
        cv2.putText(output, "Ponto clicado", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return output
