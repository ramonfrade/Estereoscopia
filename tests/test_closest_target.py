"""Testes do módulo closest_target: alvo mais próximo e profundidade no ponto."""

import numpy as np

from closest_target import (
    DepthSmoother,
    calib_left_xy_to_display,
    closest_matched_face,
    depth_at_point,
    depth_from_disparity_px,
    find_closest_target,
)
from face_detection import FaceCenter


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


def test_find_closest_target_ignores_saturated_blob():
    """SGBM satura no teto da faixa: esse blob NÃO é o alvo (distância falsa ~38 cm)."""
    disparity = np.zeros((100, 100), dtype=np.float32)
    disparity[11:30, 11:30] = 256.0  # saturação da faixa (artefato)
    disparity[61:80, 61:80] = 40.0  # objeto real

    cx, cy = find_closest_target(disparity, max_disparity=255.0)

    assert (cx, cy) == (70, 70)


def test_find_closest_target_ignores_invalid_left_band():
    """Colunas x < numDisparities não têm match válido no SGBM."""
    disparity = np.zeros((100, 100), dtype=np.float32)
    disparity[41:60, 5:25] = 80.0  # banda inválida (esquerda)
    disparity[41:60, 70:90] = 30.0  # região válida

    cx, cy = find_closest_target(disparity, margin=0.0, min_x=50)

    assert (cx, cy) == (80, 50)


def test_depth_at_point_none_when_only_saturated():
    q = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 100.0],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    disparity = np.full((50, 50), 256.0, dtype=np.float32)

    assert depth_at_point(disparity, q, unit_to_cm=1.0, x=25, y=25, max_disparity=255.0) is None


def test_depth_at_point_real_q_varies_with_disparity():
    """Q da calibração: Z = f*B/d. Se d dobra, a distância cai pela metade — nunca um valor único."""
    q = np.array(
        [
            [1.0, 0.0, 0.0, 1875.958],
            [0.0, 1.0, 0.0, -105.511],
            [0.0, 0.0, 0.0, 640.950],
            [0.0, 0.0, -0.288949, 0.0],
        ],
        dtype=np.float64,
    )
    unit_to_cm = 15.0 / 3.4608165202330667
    far = np.full((50, 50), 40.0, dtype=np.float32)
    near = np.full((50, 50), 80.0, dtype=np.float32)

    z_far = depth_at_point(far, q, unit_to_cm, x=25, y=25)
    z_near = depth_at_point(near, q, unit_to_cm, x=25, y=25)

    assert z_far is not None and z_near is not None
    assert z_far > z_near
    assert abs(z_far / z_near - 2.0) < 0.05


def _q_z_equals_100_over_d() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 100.0],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_depth_from_disparity_px_inverse_of_d():
    q = _q_z_equals_100_over_d()
    assert depth_from_disparity_px(10.0, q, 1.0) == 10.0
    assert depth_from_disparity_px(20.0, q, 1.0) == 5.0


def test_closest_matched_face_picks_larger_disparity():
    """Aproximar aumenta |x_esq - x_dir| e deve DIMINUIR a distância, não crescer."""
    q = _q_z_equals_100_over_d()
    faces_left = [
        FaceCenter(x=300.0, y=200.0, score=1.0, bbox=(0, 0, 10, 10)),  # d=80 → 1.25 cm
        FaceCenter(x=400.0, y=80.0, score=1.0, bbox=(0, 0, 10, 10)),  # d=20 → 5 cm
    ]
    faces_right = [
        FaceCenter(x=220.0, y=202.0, score=1.0, bbox=(0, 0, 10, 10)),
        FaceCenter(x=380.0, y=81.0, score=1.0, bbox=(0, 0, 10, 10)),
    ]

    hit = closest_matched_face(faces_left, faces_right, q, unit_to_cm=1.0)

    assert hit is not None
    depth_cm, face, disp = hit
    assert face.x == 300.0
    assert disp == 80.0
    assert depth_cm == 1.25


def test_closest_matched_face_none_without_pairs():
    q = _q_z_equals_100_over_d()
    left = [FaceCenter(x=100.0, y=50.0, score=1.0, bbox=(0, 0, 10, 10))]
    right = [FaceCenter(x=110.0, y=400.0, score=1.0, bbox=(0, 0, 10, 10))]  # dy grande
    assert closest_matched_face(left, right, q, 1.0, max_dy=40.0) is None


def test_calib_left_xy_to_display_shifts_when_swapped():
    assert calib_left_xy_to_display(200, 100, 40.0, swapped=True) == (160, 100)
    assert calib_left_xy_to_display(200, 100, 40.0, swapped=False) == (200, 100)


def test_depth_smoother_ema():
    s = DepthSmoother(alpha=0.5)
    assert s.update(10.0) == 10.0
    assert s.update(20.0) == 15.0
    s.update(None)
    assert s.value == 15.0
