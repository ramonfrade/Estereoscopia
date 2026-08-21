"""Estratégias de estimativa de profundidade (face vs SGBM)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

import triangulation as tri
from depth_config import DepthConfig


class DepthStrategy(ABC):
    @abstractmethod
    def estimate(
        self,
        display_left: np.ndarray,
        display_right: np.ndarray,
        rect_calib_left: np.ndarray,
        rect_calib_right: np.ndarray,
        sample_x: int | None = None,
        sample_y: int | None = None,
    ) -> dict:
        ...


class FaceDepthStrategy(DepthStrategy):
    """Mesma abordagem do src_old/stereoVision.py: MediaPipe + triangulação."""

    def __init__(self, config: DepthConfig, detector=None):
        from face_detection import FaceDetector

        self.config = config
        self._detector = detector or FaceDetector(config.face_min_confidence)
        self._owns_detector = detector is None

    def estimate(
        self,
        display_left: np.ndarray,
        display_right: np.ndarray,
        rect_calib_left: np.ndarray,
        rect_calib_right: np.ndarray,
        sample_x: int | None = None,
        sample_y: int | None = None,
    ) -> dict:
        left_vis = display_left.copy()
        right_vis = display_right.copy()

        center_left, _ = self._detector.process_with_draw(left_vis)
        center_right, _ = self._detector.process_with_draw(right_vis)

        h, w = display_left.shape[:2]
        tracking_lost = center_left is None or center_right is None

        if tracking_lost:
            cx, cy = w // 2, h // 2
            return {
                "display_left": left_vis,
                "display_right": right_vis,
                "disparity": np.zeros((h, w), dtype=np.float32),
                "depth_cm": None,
                "disparity_px": 0.0,
                "sample_point": {"x": cx, "y": cy},
                "track_mode": "lost",
            }

        # Ordem de argumentos igual ao src_old: (pt_v2, pt_v0, frame_v2, frame_v0)
        depth_cm = tri.find_depth(
            (center_left.x, center_left.y),
            (center_right.x, center_right.y),
            display_left,
            display_right,
            self.config.baseline_cm,
            self.config.focal_mm,
            self.config.fov_deg,
        )
        disp_px = abs(center_right.x - center_left.x)
        mid_x = int((center_left.x + center_right.x) / 2)
        mid_y = int((center_left.y + center_right.y) / 2)

        if depth_cm == float("inf"):
            depth_cm = None

        return {
            "display_left": left_vis,
            "display_right": right_vis,
            "disparity": np.zeros((h, w), dtype=np.float32),
            "depth_cm": depth_cm,
            "disparity_px": disp_px,
            "sample_point": {"x": mid_x, "y": mid_y},
            "track_mode": "face",
        }

    def close(self) -> None:
        if self._owns_detector:
            self._detector.close()


class SgbmDepthStrategy(DepthStrategy):
    """Disparidade densa via StereoSGBM + matriz Q (modo experimental)."""

    def __init__(self, stereo_matcher, q_matrix: np.ndarray, unit_to_cm: float):
        self._stereo = stereo_matcher
        self._q_matrix = q_matrix
        self._unit_to_cm = unit_to_cm
        self._prev_gray: np.ndarray | None = None

    @staticmethod
    def find_closest_point(
        disparity: np.ndarray, mask: np.ndarray | None = None, margin: float = 0.08
    ) -> tuple[int, int]:
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
        half = window // 2
        y1, y2 = max(0, y - half), min(disparity.shape[0], y + half + 1)
        x1, x2 = max(0, x - half), min(disparity.shape[1], x + half + 1)
        patch = disparity[y1:y2, x1:x2]
        valid = patch[patch > 0.5]
        if valid.size == 0:
            return None, 0.0

        d = float(np.median(valid))
        q = self._q_matrix
        w = q[3, 2] * d + q[3, 3]
        if abs(w) < 1e-6:
            return None, d
        z_units = q[2, 3] / w
        depth_cm = abs(z_units) * self._unit_to_cm
        if depth_cm <= 0 or depth_cm > 1_000:
            return None, d
        return depth_cm, d

    def estimate(
        self,
        display_left: np.ndarray,
        display_right: np.ndarray,
        rect_calib_left: np.ndarray,
        rect_calib_right: np.ndarray,
        sample_x: int | None = None,
        sample_y: int | None = None,
    ) -> dict:
        # SGBM na ordem da calibração (a mesma da matriz Q). A ordem física
        # inverte d: aproximar faz a distância exibida crescer.
        gray_left = cv2.cvtColor(rect_calib_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rect_calib_right, cv2.COLOR_BGR2GRAY)

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
            "display_left": display_left,
            "display_right": display_right,
            "disparity": disparity,
            "depth_cm": depth_cm,
            "disparity_px": disp_value,
            "sample_point": {"x": int(cx), "y": int(cy)},
            "track_mode": track_mode,
        }
