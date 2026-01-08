"""
Shared pytest fixtures and test data for ivt_filter tests.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pytest


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent.parent / "test_data"


@pytest.fixture
def archive_data_dir(test_data_dir):
    """Return path to archive test data."""
    return test_data_dir / "archive"


@pytest.fixture
def simple_eye_tracking_data():
    """Create simple synthetic eye-tracking data for unit tests."""
    n_samples = 100
    data = {
        'time_ms': np.arange(n_samples),
        'gaze_left_x_mm': np.linspace(100, 110, n_samples),
        'gaze_left_y_mm': np.linspace(200, 210, n_samples),
        'gaze_right_x_mm': np.linspace(100, 110, n_samples),
        'gaze_right_y_mm': np.linspace(200, 210, n_samples),
        'eye_left_x_mm': np.zeros(n_samples),
        'eye_left_y_mm': np.zeros(n_samples),
        'eye_left_z_mm': np.full(n_samples, 600.0),
        'eye_right_x_mm': np.zeros(n_samples),
        'eye_right_y_mm': np.zeros(n_samples),
        'eye_right_z_mm': np.full(n_samples, 600.0),
        'validity_left': np.zeros(n_samples, dtype=int),
        'validity_right': np.zeros(n_samples, dtype=int),
    }
    return pd.DataFrame(data)


@pytest.fixture
def saccade_fixation_data():
    """Create synthetic data with clear fixation and saccade patterns."""
    n_samples = 200
    time_ms = np.arange(n_samples)
    
    # First 50: slow fixation (1 deg/s ~ 1mm/frame at 600mm distance)
    gaze_x = np.concatenate([
        np.ones(50) * 100,          # Fixation 1
        np.linspace(100, 150, 20),  # Saccade to right
        np.ones(80) * 150,          # Fixation 2
        np.linspace(150, 100, 20),  # Saccade back
        np.ones(30) * 100,          # Fixation 3
    ])
    
    gaze_y = np.ones(n_samples) * 200
    
    data = {
        'time_ms': time_ms,
        'gaze_left_x_mm': gaze_x,
        'gaze_left_y_mm': gaze_y,
        'gaze_right_x_mm': gaze_x,
        'gaze_right_y_mm': gaze_y,
        'eye_left_x_mm': np.zeros(n_samples),
        'eye_left_y_mm': np.zeros(n_samples),
        'eye_left_z_mm': np.full(n_samples, 600.0),
        'eye_right_x_mm': np.zeros(n_samples),
        'eye_right_y_mm': np.zeros(n_samples),
        'eye_right_z_mm': np.full(n_samples, 600.0),
        'validity_left': np.zeros(n_samples, dtype=int),
        'validity_right': np.zeros(n_samples, dtype=int),
    }
    return pd.DataFrame(data)


@pytest.fixture
def data_with_gaps():
    """Create data with missing/invalid samples."""
    n_samples = 100
    data = {
        'time_ms': np.arange(n_samples),
        'gaze_left_x_mm': np.linspace(100, 110, n_samples),
        'gaze_left_y_mm': np.linspace(200, 210, n_samples),
        'gaze_right_x_mm': np.linspace(100, 110, n_samples),
        'gaze_right_y_mm': np.linspace(200, 210, n_samples),
        'eye_left_x_mm': np.zeros(n_samples),
        'eye_left_y_mm': np.zeros(n_samples),
        'eye_left_z_mm': np.full(n_samples, 600.0),
        'eye_right_x_mm': np.zeros(n_samples),
        'eye_right_y_mm': np.zeros(n_samples),
        'eye_right_z_mm': np.full(n_samples, 600.0),
        'validity_left': np.zeros(n_samples, dtype=int),
        'validity_right': np.zeros(n_samples, dtype=int),
    }
    df = pd.DataFrame(data)
    
    # Add gaps: mark indices 20-25 and 60-65 as invalid
    df.loc[20:25, 'validity_left'] = 1
    df.loc[20:25, 'validity_right'] = 1
    df.loc[60:65, 'validity_left'] = 1
    df.loc[60:65, 'validity_right'] = 1
    
    return df


@pytest.fixture
def mixed_eye_validity_data():
    """Create data where only one eye is valid at certain times."""
    n_samples = 100
    data = {
        'time_ms': np.arange(n_samples),
        'gaze_left_x_mm': np.linspace(100, 110, n_samples),
        'gaze_left_y_mm': np.linspace(200, 210, n_samples),
        'gaze_right_x_mm': np.linspace(100, 110, n_samples),
        'gaze_right_y_mm': np.linspace(200, 210, n_samples),
        'eye_left_x_mm': np.zeros(n_samples),
        'eye_left_y_mm': np.zeros(n_samples),
        'eye_left_z_mm': np.full(n_samples, 600.0),
        'eye_right_x_mm': np.zeros(n_samples),
        'eye_right_y_mm': np.zeros(n_samples),
        'eye_right_z_mm': np.full(n_samples, 600.0),
        'validity_left': np.zeros(n_samples, dtype=int),
        'validity_right': np.zeros(n_samples, dtype=int),
    }
    df = pd.DataFrame(data)
    
    # First 30: only left valid
    df.loc[0:30, 'validity_right'] = 1
    # Next 40: only right valid
    df.loc[31:70, 'validity_left'] = 1
    # Last 30: both valid
    
    return df


@pytest.fixture
def real_data_sample(archive_data_dir):
    """Load a real sample from archive if available."""
    archive_files = list(archive_data_dir.glob("*.tsv"))
    if not archive_files:
        pytest.skip("No archive data available")
    
    # Use first available file
    sample_file = archive_files[0]
    try:
        df = pd.read_csv(sample_file, sep="\t", decimal=",", low_memory=False)
        return df
    except Exception as e:
        pytest.skip(f"Could not load archive data: {e}")
