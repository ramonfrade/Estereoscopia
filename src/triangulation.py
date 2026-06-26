"""Estimativa de profundidade por disparidade horizontal entre dois pontos correspondentes."""

import numpy as np


def find_depth(right_center, left_center, frame_right, frame_left, baseline_cm, f_mm, fov_deg):
    """
    Calcula profundidade aproximada (cm) a partir de centros correspondentes nas imagens retificadas.

    baseline_cm: distância entre câmeras em centímetros
    f_mm: distância focal da lente em milímetros (aproximação; preferir matriz Q quando disponível)
    fov_deg: campo de visão horizontal em graus
    """
    _height_right, width_right = frame_right.shape[:2]
    _height_left, width_left = frame_left.shape[:2]

    if width_right != width_left:
        raise ValueError("Frames esquerdo e direito devem ter a mesma largura.")

    f_pixel = (width_right * 0.5) / np.tan(np.deg2rad(fov_deg * 0.5))
    x_right = float(right_center[0])
    x_left = float(left_center[0])

    disparity = abs(x_left - x_right)
    if disparity < 1.0:
        return float("inf")

    # baseline em cm, resultado em cm
    z_depth = (baseline_cm * f_pixel) / disparity
    return abs(z_depth)
