import sys
import cv2
import numpy as np
import time


def find_depth(right_point, left_point, frame_right, frame_left, baseline_cm, f_mm, alpha):

    # CONVERT FOCAL LENGTH F FROM [MM] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left,  width_left,  depth_left  = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)

    else:
        raise ValueError("Frames esquerdo e direito devem ter a mesma largura.")

    x_right = right_point[0]
    x_left  = left_point[0]

    # CALCULATE THE DISPARITY:
    disparity = x_left - x_right

    # CALCULATE DEPTH Z:
    z_depth = (baseline_cm * f_pixel) / disparity
    
    return abs(z_depth)
