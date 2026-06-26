from pathlib import Path

import cv2
import numpy as np
import pytest

import calibration

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def stereo_map_exists():
    if not (PROJECT_ROOT / "stereoMap.xml").exists():
        pytest.skip("stereoMap.xml ausente — rode stereo_calibration.py")


@pytest.fixture
def calib_pair():
    left = PROJECT_ROOT / "images" / "stereoLeft" / "imageL0.png"
    right = PROJECT_ROOT / "images" / "stereoRight" / "imageR0.png"
    if not left.exists() or not right.exists():
        pytest.skip("imagens de calibração ausentes")
    img_l = cv2.imread(str(left))
    img_r = cv2.imread(str(right))
    assert img_l is not None and img_r is not None
    return img_l, img_r


def test_undistort_rectify_shape(stereo_map_exists, calib_pair):
    img_l, img_r = calib_pair
    rect_l, rect_r = calibration.undistort_rectify(img_l, img_r)
    assert rect_l.shape == img_l.shape
    assert rect_r.shape == img_r.shape
    assert rect_l.dtype == np.uint8


def test_rectify_reduces_black_border_ratio(stereo_map_exists, calib_pair):
    img_l, img_r = calib_pair
    rect_l, rect_r = calibration.undistort_rectify(img_l, img_r)
    gray = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    black_ratio = (gray < 5).mean()
    assert black_ratio < 0.5
