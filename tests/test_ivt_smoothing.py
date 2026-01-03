import pandas as pd

from ivt_filter.velocity import (apply_ivt_post_smoothing, _rebuild_ivt_event_columns,
                      IVTClassifierConfig)


def test_post_smoothing_reclassifies_short_saccade_inside_fixation():
    # Construct artificial sample-level labels and times
    times = [0, 10, 20, 30, 40, 50, 60]
    # initial labels: two fixations around a 3-sample saccade
    labels = ["Fixation", "Fixation", "Saccade", "Saccade", "Saccade", "Fixation", "Fixation"]

    df = pd.DataFrame({"time_ms": times, "ivt_sample_type": labels})
    df = _rebuild_ivt_event_columns(df, label_col="ivt_sample_type")

    cfg = IVTClassifierConfig(post_smoothing_ms=25.0)
    df_smoothed = apply_ivt_post_smoothing(df, cfg, time_col="time_ms")

    # The saccade duration is 20 ms (20 -> 40 ms), threshold is 25 ms, so it
    # should be reclassified to Fixation since it is bounded by fixations.
    assert all(s == "Fixation" for s in df_smoothed.loc[2:4, "ivt_sample_type"].tolist())

    # Event indices should be recomputed to reflect the single fixation event
    assert df_smoothed.loc[0, "ivt_event_type"] == "Fixation"
    assert df_smoothed.loc[6, "ivt_event_type"] == "Fixation"
    assert df_smoothed.loc[0, "ivt_event_index"] == df_smoothed.loc[6, "ivt_event_index"]
