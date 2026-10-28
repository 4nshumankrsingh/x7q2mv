import pandas as pd

from app.services.ingest import (
    actuals_to_frame,
    build_report,
    forecasts_to_frame,
    restrict_to_window,
)


def test_actuals_are_deduped_sorted_utc_and_null_free() -> None:
    rows = [
        {"startTime": "2026-01-01T00:30:00Z", "generation": 90},
        {"startTime": "2026-01-01T00:00:00Z", "generation": 100},
        {"startTime": "2026-01-01T00:00:00Z", "generation": 110},  # duplicate: keep last
        {"startTime": "2026-01-01T01:00:00Z", "generation": None},  # null: dropped
    ]
    df, dropped = actuals_to_frame(rows)
    assert dropped == 1
    assert list(df["generation"]) == [110.0, 90.0]
    assert df["startTime"].is_monotonic_increasing
    assert str(df["startTime"].dt.tz) == "UTC"


def test_forecasts_dedupe_on_target_and_publish_time() -> None:
    target = "2026-01-02T00:00:00Z"
    rows = [
        {"publishTime": "2026-01-01T06:00:00Z", "startTime": target, "generation": 10},
        {"publishTime": "2026-01-01T06:00:00Z", "startTime": target, "generation": 12},
        {"publishTime": "2026-01-01T07:00:00Z", "startTime": target, "generation": 14},
    ]
    df, dropped = forecasts_to_frame(rows)
    assert dropped == 0
    assert list(df["generation"]) == [12.0, 14.0]


def test_restrict_to_window_is_half_open() -> None:
    times = [
        "2025-12-31T23:30:00Z",
        "2026-01-01T00:00:00Z",
        "2026-01-31T23:30:00Z",
        "2026-02-01T00:00:00Z",
    ]
    df, _ = actuals_to_frame([{"startTime": t, "generation": 1} for t in times])
    out = restrict_to_window(
        df, pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-02-01", tz="UTC")
    )
    assert out["startTime"].dt.strftime("%Y-%m-%dT%H:%M").tolist() == [
        "2026-01-01T00:00",
        "2026-01-31T23:30",
    ]


def test_report_finds_missing_actuals_and_horizons() -> None:
    start = pd.Timestamp("2026-01-01T00:00", tz="UTC")
    end = pd.Timestamp("2026-01-01T02:00", tz="UTC")  # 4 half-hours
    actuals, _ = actuals_to_frame(
        [
            {"startTime": "2026-01-01T00:00:00Z", "generation": 1},
            {"startTime": "2026-01-01T01:00:00Z", "generation": 1},
            {"startTime": "2026-01-01T01:30:00Z", "generation": 1},
        ]
    )
    forecasts, _ = forecasts_to_frame(
        [
            {"publishTime": "2025-12-31T12:00:00Z", "startTime": "2026-01-01T00:00:00Z", "generation": 5},
            {"publishTime": "2025-12-31T12:00:00Z", "startTime": "2026-01-01T00:30:00Z", "generation": 5},
            {"publishTime": "2025-12-31T18:00:00Z", "startTime": "2026-01-01T00:30:00Z", "generation": 6},
        ]
    )
    report = build_report(actuals, forecasts, start, end)
    assert report["expected_half_hours"] == 4
    assert report["missing_actual_count"] == 1
    assert report["targets_with_forecast"] == 2
    assert report["targets_without_forecast"] == 2
    assert report["versions_per_target"]["max"] == 2
    assert report["max_horizon_h"]["max"] == 12.5