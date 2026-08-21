"""Testes do módulo closest_target: alvo mais próximo e profundidade no ponto."""

import numpy as np

from closest_target import depth_at_point, find_closest_target


def test_find_closest_target_returns_centroid_of_nearest_blob():
    disparity = np.zeros((100, 100), dtype=np.float32)
    disparity[11:30, 11:30] = 50.0  # blob próximo (disparidade alta)
    disparity[61:80, 61:80] = 20.0  # blob mais distante

    cx, cy = find_closest_target(disparity)

    assert (cx, cy) == (20, 20)


def test_find_closest_target_none_when_no_valid_disparity():
    disparity = np.zeros((100, 100), dtype=np.float32)

    assert find_closest_target(disparity) is None


def test_find_closest_target_ignores_border_margin():
    disparity = np.zeros((100, 100), dtype=np.float32)
    disparity[0:5, 0:5] = 100.0  # totalmente na margem -> descartado
    disparity[41:60, 41:60] = 30.0

    cx, cy = find_closest_target(disparity, margin=0.08)

    assert (cx, cy) == (50, 50)


def test_find_closest_target_min_area_filters_speckle():
    disparity = np.zeros((100, 100), dtype=np.float32)
    disparity[5, 5] = 90.0  # pixel isolado (speckle) — menor que a área mínima
    disparity[41:60, 41:60] = 30.0

    cx, cy = find_closest_target(disparity, margin=0.0, min_area_pct=0.002)

    assert (cx, cy) == (50, 50)


def test_depth_at_point_uses_q_matrix():
    q = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 100.0],  # q[2,3] = f*b
            [0.0, 0.0, -1.0, 0.0],  # w = -d
        ],
        dtype=np.float64,
    )
    disparity = np.full((50, 50), 10.0, dtype=np.float32)

    depth = depth_at_point(disparity, q, unit_to_cm=1.0, x=25, y=25)

    assert depth == 10.0


def test_depth_at_point_none_when_no_valid_disparity():
    q = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 100.0],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    disparity = np.zeros((50, 50), dtype=np.float32)

    assert depth_at_point(disparity, q, unit_to_cm=1.0, x=25, y=25) is None
