import numpy as np
import pytest

import triangulation as tri


def _blank_frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_find_depth_known_disparity():
    w, h = 640, 480
    frame_r = _blank_frame(w, h)
    frame_l = _blank_frame(w, h)
    baseline = 7.0
    f_mm = 8.0
    fov = 56.6
    f_pixel = (w * 0.5) / np.tan(np.deg2rad(fov * 0.5))

    x_right = 280.0
    x_left = 300.0
    disparity = 20.0
    expected = (baseline * f_pixel) / disparity

    depth = tri.find_depth(
        (x_right, 240.0),
        (x_left, 240.0),
        frame_r,
        frame_l,
        baseline,
        f_mm,
        fov,
    )
    assert depth == pytest.approx(expected, rel=1e-4)


def test_find_depth_zero_disparity_returns_inf():
    frame = _blank_frame()
    depth = tri.find_depth((100, 240), (100, 240), frame, frame, 7, 8, 56.6)
    assert depth == float("inf")


def test_find_depth_width_mismatch_raises():
    frame_a = _blank_frame(640, 480)
    frame_b = _blank_frame(320, 240)
    with pytest.raises(ValueError, match="mesma largura"):
        tri.find_depth((10, 10), (20, 10), frame_a, frame_b, 7, 8, 56.6)
