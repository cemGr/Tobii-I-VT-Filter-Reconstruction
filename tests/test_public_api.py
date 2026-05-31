"""Smoke tests for the public simple API imports."""


def test_import_process_eye_tracking() -> None:
    from ivt_filter.simple_api import process_eye_tracking

    assert callable(process_eye_tracking)


def test_import_process_eye_tracking_adaptive() -> None:
    from ivt_filter.simple_api import process_eye_tracking_adaptive

    assert callable(process_eye_tracking_adaptive)


def test_import_get_statistics() -> None:
    from ivt_filter.simple_api import get_statistics

    assert callable(get_statistics)


def test_import_print_statistics() -> None:
    from ivt_filter.simple_api import print_statistics

    assert callable(print_statistics)
