import cv2
import numpy as np

from depth_config import DepthConfig
from depth_strategies import FaceDepthStrategy, SgbmDepthStrategy


class FakeDetector:
    def process_with_draw(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if frame_bgr[0, 0, 0] == 1:
            return type("C", (), {"x": 300.0, "y": 240.0, "score": 0.9, "bbox": (0, 0, 10, 10)})(), None
        return type("C", (), {"x": 320.0, "y": 240.0, "score": 0.9, "bbox": (0, 0, 10, 10)})(), None

    def close(self):
        pass


def test_face_strategy_returns_depth():
    left = np.full((480, 640, 3), 1, dtype=np.uint8)
    right = np.full((480, 640, 3), 2, dtype=np.uint8)
    strat = FaceDepthStrategy(DepthConfig(), detector=FakeDetector())
    result = strat.estimate(left, right, left, right)
    assert result["track_mode"] == "face"
    assert result["depth_cm"] is not None
    assert result["depth_cm"] > 0
    assert result["disparity_px"] == 20.0


def test_face_strategy_tracking_lost():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    class NoFace:
        def process_with_draw(self, _):
            return None, None

        def close(self):
            pass

    strat = FaceDepthStrategy(DepthConfig(), detector=NoFace())
    result = strat.estimate(blank, blank, blank, blank)
    assert result["track_mode"] == "lost"
    assert result["depth_cm"] is None


class FakeMatcher:
    def __init__(self):
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def compute(self, gray_left, gray_right):
        self.calls.append((gray_left.copy(), gray_right.copy()))
        h, w = gray_left.shape
        return np.full((h, w), 16 * 10, dtype=np.int16)


def test_sgbm_strategy_uses_calib_order_for_q():
    """Q vem do stereoRectify na ordem de calibração; SGBM tem de usar o mesmo par.

    Ordem física (display) com physicalLeft != calibLeft inverte d: aproximar
    *aumenta* a distância exibida e o mapa não corresponde à matriz Q.
    """
    rect_l = np.zeros((20, 30, 3), dtype=np.uint8)  # calib left (video0)
    rect_r = np.full((20, 30, 3), 255, dtype=np.uint8)  # calib right (video2)
    display_l, display_r = rect_r, rect_l  # ordem física TROCADA

    matcher = FakeMatcher()
    strat = SgbmDepthStrategy(matcher, np.eye(4), unit_to_cm=1.0)
    strat.estimate(display_l, display_r, rect_l, rect_r)

    assert matcher.calls, "matcher nunca foi chamado"
    gray_l, gray_r = matcher.calls[0]
    assert np.array_equal(gray_l, cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY))
    assert np.array_equal(gray_r, cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY))
