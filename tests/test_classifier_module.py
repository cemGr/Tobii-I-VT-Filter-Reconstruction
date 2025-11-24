import pandas as pd

from ivt.classifier import IVTClassifier


def test_classify_tolerates_string_velocity():
    df = pd.DataFrame({"velocity_deg_per_sec": ["10.5", "35", None, "nan"]})

    classified = IVTClassifier().classify(df)

    assert list(classified["ivt_sample_type"]) == ["Fixation", "Saccade", "Unclassified", "Unclassified"]
    assert classified["ivt_event_index"].tolist() == [1, 2, pd.NA, pd.NA]
