import math

import numpy as np

from ivt_filter.velocity_calculation import Ray3DGazeDir, Ray3DAngle


def test_ray3d_gaze_dir_right_angle():
    calc = Ray3DGazeDir()
    deg = calc.calculate_visual_angle(0, 0, 0, 0, None, None, None, dir1=(1, 0, 0), dir2=(0, 1, 0))
    assert math.isclose(deg, 90.0, rel_tol=1e-6, abs_tol=1e-6)


def test_ray3d_gaze_dir_clips_and_normalizes():
    calc = Ray3DGazeDir()
    # Non-normalized inputs and nearly opposite directions should clip to 180°
    dir1 = (10, 0, 0)
    dir2 = (-10 + 1e-9, 0, 0)
    deg = calc.calculate_visual_angle(0, 0, 0, 0, None, None, None, dir1=dir1, dir2=dir2)
    assert math.isclose(deg, 180.0, rel_tol=1e-6, abs_tol=1e-6)


def test_ray3d_angle_simple_geometry():
    # Eye at (0,0,600), gaze points 10 mm apart on screen; angle ~0.955°
    calc = Ray3DAngle()
    deg = calc.calculate_visual_angle(
        x1_mm=0.0,
        y1_mm=0.0,
        x2_mm=10.0,
        y2_mm=0.0,
        eye_x_mm=0.0,
        eye_y_mm=0.0,
        eye_z_mm=600.0,
    )
    assert math.isclose(deg, 0.95484, rel_tol=1e-3, abs_tol=1e-3)
