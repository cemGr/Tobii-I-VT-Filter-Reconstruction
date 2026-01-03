import pandas as pd
from ivt_filter.classification import apply_ivt_classifier

def test_eyesnotfound_labeling():
    # Simulate DataFrame with both eyes invalid
    df = pd.DataFrame({
        'velocity_deg_per_sec': [None, float('nan'), 10, 100],
        'left_eye_valid': [False, False, True, False],
        'right_eye_valid': [False, False, False, True],
    })
    # Should label first two as EyesNotFound, third as Fixation, fourth as Saccade
    result = apply_ivt_classifier(df)
    labels = result['ivt_sample_type'].tolist()
    assert labels[0] == 'EyesNotFound', f"Expected EyesNotFound, got {labels[0]}"
    assert labels[1] == 'EyesNotFound', f"Expected EyesNotFound, got {labels[1]}"
    assert labels[2] == 'Fixation', f"Expected Fixation, got {labels[2]}"
    assert labels[3] == 'Saccade', f"Expected Saccade, got {labels[3]}"

def test_unclassified_labeling():
    # Simulate DataFrame with one eye valid, velocity invalid
    df = pd.DataFrame({
        'velocity_deg_per_sec': [None, float('nan'), 10, 100],
        'left_eye_valid': [True, False, True, True],
        'right_eye_valid': [False, True, True, True],
    })
    # Should label first two as Unclassified, third as Fixation, fourth as Saccade
    result = apply_ivt_classifier(df)
    labels = result['ivt_sample_type'].tolist()
    assert labels[0] == 'Unclassified', f"Expected Unclassified, got {labels[0]}"
    assert labels[1] == 'Unclassified', f"Expected Unclassified, got {labels[1]}"
    assert labels[2] == 'Fixation', f"Expected Fixation, got {labels[2]}"
    assert labels[3] == 'Saccade', f"Expected Saccade, got {labels[3]}"
