import numpy as np

from depth_config import DepthConfig
from depth_strategies import FaceDepthStrategy


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
