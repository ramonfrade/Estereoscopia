import numpy as np
import pytest

from face_detection import FaceDetector, bbox_center


def test_bbox_center():
    cx, cy, (bx, by, bw, bh) = bbox_center(0.1, 0.2, 0.3, 0.4, 1000, 800)
    assert bx == 100
    assert by == 160
    assert bw == 300
    assert bh == 320
    assert cx == 100 + 150
    assert cy == 160 + 160


pytest.importorskip("mediapipe")


def test_detect_all_with_draw_empty_on_blank_frame():
    detector = FaceDetector()
    try:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        faces = detector.detect_all_with_draw(frame)
        assert faces == []
    finally:
        detector.close()
