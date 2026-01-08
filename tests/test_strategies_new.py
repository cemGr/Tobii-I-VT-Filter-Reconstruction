"""
Tests for velocity strategies that exist in the project.
"""
import pytest
import math
import numpy as np
from ivt_filter.strategies.velocity_calculation import (
    Olsen2DApproximation,
    Ray3DAngle,
    Ray3DGazeDir,
)


class TestOlsen2D:
    """Tests for Olsen 2D strategy."""
    
    def test_olsen2d_simple_movement(self):
        """Test basic movement calculation."""
        calc = Olsen2DApproximation()
        
        # 10mm movement at 600mm distance
        angle = calc.calculate_visual_angle(
            x1_mm=0, y1_mm=0,
            x2_mm=10, y2_mm=0,
            eye_x_mm=0, eye_y_mm=0, eye_z_mm=600
        )
        
        # Should be approximately atan(10/600) in degrees
        expected = math.degrees(math.atan(10/600))
        assert abs(angle - expected) < 0.01
    
    def test_olsen2d_zero_movement(self):
        """Zero movement should give zero angle."""
        calc = Olsen2DApproximation()
        
        angle = calc.calculate_visual_angle(
            x1_mm=100, y1_mm=200,
            x2_mm=100, y2_mm=200,
            eye_x_mm=0, eye_y_mm=0, eye_z_mm=600
        )
        
        assert angle == 0.0
    
    def test_olsen2d_diagonal_movement(self):
        """Test diagonal movement."""
        calc = Olsen2DApproximation()
        
        angle = calc.calculate_visual_angle(
            x1_mm=0, y1_mm=0,
            x2_mm=10, y2_mm=10,
            eye_x_mm=0, eye_y_mm=0, eye_z_mm=600
        )
        
        # Should be larger than single-axis movement
        dist = math.sqrt(10**2 + 10**2)
        expected = math.degrees(math.atan(dist/600))
        assert abs(angle - expected) < 0.01


class TestRay3D:
    """Tests for Ray 3D strategy."""
    
    def test_ray3d_perpendicular_movement(self):
        """Test Ray 3D calculation."""
        calc = Ray3DAngle()
        
        angle = calc.calculate_visual_angle(
            x1_mm=0, y1_mm=0,
            x2_mm=10, y2_mm=0,
            eye_x_mm=0, eye_y_mm=0, eye_z_mm=600
        )
        
        # Should return positive angle
        assert angle > 0
        assert angle < 2.0  # Should be small angle
    
    def test_ray3d_zero_movement(self):
        """Zero movement should give zero angle."""
        calc = Ray3DAngle()
        
        angle = calc.calculate_visual_angle(
            x1_mm=100, y1_mm=200,
            x2_mm=100, y2_mm=200,
            eye_x_mm=0, eye_y_mm=0, eye_z_mm=600
        )
        
        assert angle == 0.0


class TestRay3DGazeDir:
    """Tests for Ray 3D with gaze direction."""
    
    def test_ray3d_gaze_dir_90_degrees(self):
        """Perpendicular directions should give 90 degrees."""
        calc = Ray3DGazeDir()
        
        angle = calc.calculate_visual_angle(
            x1_mm=0, y1_mm=0,
            x2_mm=0, y2_mm=0,
            eye_x_mm=0, eye_y_mm=0, eye_z_mm=0,
            dir1=(1, 0, 0),
            dir2=(0, 1, 0)
        )
        
        assert math.isclose(angle, 90.0, abs_tol=1e-6)
    
    def test_ray3d_gaze_dir_zero_angle(self):
        """Same direction should give zero angle."""
        calc = Ray3DGazeDir()
        
        angle = calc.calculate_visual_angle(
            x1_mm=0, y1_mm=0,
            x2_mm=0, y2_mm=0,
            eye_x_mm=0, eye_y_mm=0, eye_z_mm=0,
            dir1=(1, 0, 0),
            dir2=(1, 0, 0)
        )
        
        assert math.isclose(angle, 0.0, abs_tol=1e-6)
    
    def test_ray3d_gaze_dir_180_degrees(self):
        """Opposite directions should give 180 degrees."""
        calc = Ray3DGazeDir()
        
        angle = calc.calculate_visual_angle(
            x1_mm=0, y1_mm=0,
            x2_mm=0, y2_mm=0,
            eye_x_mm=0, eye_y_mm=0, eye_z_mm=0,
            dir1=(1, 0, 0),
            dir2=(-1, 0, 0)
        )
        
        assert math.isclose(angle, 180.0, abs_tol=1e-6)
    
    def test_ray3d_gaze_dir_normalizes(self):
        """Should normalize non-unit vectors."""
        calc = Ray3DGazeDir()
        
        angle1 = calc.calculate_visual_angle(
            x1_mm=0, y1_mm=0, x2_mm=0, y2_mm=0,
            eye_x_mm=0, eye_y_mm=0, eye_z_mm=0,
            dir1=(1, 0, 0), dir2=(0, 1, 0)
        )
        
        angle2 = calc.calculate_visual_angle(
            x1_mm=0, y1_mm=0, x2_mm=0, y2_mm=0,
            eye_x_mm=0, eye_y_mm=0, eye_z_mm=0,
            dir1=(10, 0, 0), dir2=(0, 100, 0)
        )
        
        # Should be same after normalization
        assert math.isclose(angle1, angle2, abs_tol=1e-6)


class TestVelocityComparison:
    """Compare different velocity calculation methods."""
    
    def test_olsen_vs_ray3d(self):
        """Compare Olsen 2D and Ray 3D for same movement."""
        olsen = Olsen2DApproximation()
        ray3d = Ray3DAngle()
        
        angle_olsen = olsen.calculate_visual_angle(
            x1_mm=0, y1_mm=0,
            x2_mm=10, y2_mm=0,
            eye_x_mm=0, eye_y_mm=0, eye_z_mm=600
        )
        
        angle_ray3d = ray3d.calculate_visual_angle(
            x1_mm=0, y1_mm=0,
            x2_mm=10, y2_mm=0,
            eye_x_mm=0, eye_y_mm=0, eye_z_mm=600
        )
        
        # Should be close but not identical
        assert abs(angle_olsen - angle_ray3d) < 0.1
        # Ray3D typically gives slightly smaller angles
        assert angle_ray3d <= angle_olsen * 1.05
