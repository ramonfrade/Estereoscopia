import numpy as np

from depth_config import DepthMode
from depth_strategies import SgbmDepthStrategy
from stereo_engine import CameraLayout, StereoEngine, usable_num_disparities


def test_camera_layout_from_json():
    data = {
        "cameras": {
            "calibLeft": 0,
            "calibRight": 2,
            "physicalLeft": 2,
            "physicalRight": 0,
        }
    }
    layout = CameraLayout.from_calibration_json(data)
    assert layout.calib_left == 0
    assert layout.calib_right == 2
    assert layout.physical_left == 2
    assert layout.physical_right == 0


def test_rectified_for_physical_swaps_display():
    layout = CameraLayout(calib_left=0, calib_right=2, physical_left=2, physical_right=0)
    rect_l = np.full((10, 10, 3), 1, dtype=np.uint8)
    rect_r = np.full((10, 10, 3), 2, dtype=np.uint8)
    disp_l, disp_r = StereoEngine._rectified_for_physical(rect_l, rect_r, layout)
    assert disp_l[0, 0, 0] == 2
    assert disp_r[0, 0, 0] == 1


def test_find_closest_point_picks_max_disparity():
    disp = np.zeros((100, 100), dtype=np.float32)
    disp[50, 60] = 40.0
    disp[50, 30] = 10.0
    x, y = SgbmDepthStrategy.find_closest_point(disp, margin=0.0)
    assert (x, y) == (60, 50)


def test_find_closest_point_fallback_center():
    disp = np.zeros((100, 100), dtype=np.float32)
    x, y = SgbmDepthStrategy.find_closest_point(disp)
    assert (x, y) == (50, 50)


def test_sgbm_depth_at_point():
    q = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 640.95],
            [0, 0, -0.288949, 0],
        ],
        dtype=np.float64,
    )
    strat = SgbmDepthStrategy(stereo_matcher=None, q_matrix=q, unit_to_cm=2.0)
    disp = np.zeros((50, 50), dtype=np.float32)
    disp[25, 25] = 20.0
    depth, d = strat.depth_at_point(disp, 25, 25, window=5)
    assert d == 20.0
    assert depth is not None
    assert 0 < depth < 1000


def test_matcher_disparity_range_covers_close_objects():
    """Com f=641 px e baseline 15 cm, numDisparities=128 satura a ~76 cm.

    O alvo mais próximo (dedo/mão) fica 'preso' em ~76 cm; a faixa padrão
    precisa cobrir objetos bem mais próximos (>= 256 disparidades).
    """
    matcher = StereoEngine._create_stereo_matcher()
    assert matcher.getNumDisparities() >= 256


def test_usable_num_disparities_keeps_frame_center_valid():
    """SGBM anula colunas x < numDisparities. 512 em 640 px deixa o centro inválido."""
    nd = usable_num_disparities(frame_width=640, requested=512)
    assert nd % 16 == 0
    assert nd <= 640 // 2
    assert nd >= 256


def test_engine_clamps_num_disparities_to_keep_center_valid():
    eng = StereoEngine(depth_mode=DepthMode.SGBM, num_disparities=512)
    assert eng.num_disparities <= eng.frame_size[0] // 2
    assert eng.num_disparities % 16 == 0
    assert eng._stereo.getNumDisparities() == eng.num_disparities
