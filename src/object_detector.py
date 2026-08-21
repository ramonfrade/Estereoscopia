"""Detecção de objetos em fundo homogêneo (padrão do tutorial "Measure size").

Reimplementação autocontida do `object_detector.HomogeneousBgDetector` usado
pelo tutorial Pysource — sem dependência externa, pronta para pytest.
"""

from __future__ import annotations

import cv2
import numpy as np


class HomogeneousBgDetector:
    """Detecta contornos de objetos sobre um fundo de cor uniforme."""

    def detect_objects(self, frame: np.ndarray, min_area: float = 2000) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            19,
            5,
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]


def object_size_cm(cnt: np.ndarray, pixel_cm_ratio: float) -> tuple[float, float]:
    """Largura e altura do objeto em cm a partir do retângulo de área mínima."""
    if pixel_cm_ratio <= 0:
        raise ValueError("pixel_cm_ratio deve ser positivo.")
    (_, _), (w_px, h_px), _ = cv2.minAreaRect(cnt)
    return w_px / pixel_cm_ratio, h_px / pixel_cm_ratio
