import pandas as pd
import pytest

from app.services.metrics import add_band, align, error_metrics, metrics_by


def _ts(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_datetime(df[col], utc=True).dt.as_unit("ns")
    return df


def test_error_metrics_match_hand_computation() -> None:
    df = pd.DataFrame({"forecast": [110.0, 90.0, 120.0], "actual": [100.0, 100.0, 100.0]})
    m = error_metrics(df)  # errors: +10, -10, +20
    assert m["n"] == 3
    assert m["mae"] == pytest.approx(40 / 3)
    assert m["rmse"] == pytest.approx((600 / 3) ** 0.5)
    assert m["bias"] == pytest.approx(20 / 3)
    assert m["wmape"] == pytest.approx(40 / 300)


def test_error_metrics_empty_and_zero_actuals() -> None:
    assert error_metrics(pd.DataFrame({"forecast": [], "actual": []}))["mae"] is None
    zero = error_metrics(pd.DataFrame({"forecast": [5.0], "actual": [0.0]}))
    assert zero["wmape"] is None  # no division by zero
    assert zero["mae"] == 5.0


def test_align_same_start_drops_unmatched_and_signs_error() -> None:
    fc = _ts(
        pd.DataFrame(
            {
                "publishTime": ["2026-01-01T12:00:00Z"] * 2,
                "startTime": ["2026-01-02T00:00:00Z", "2026-01-02T01:00:00Z"],
                "forecast": [110.0, 90.0],
            }
        ),
        "publishTime",
        "startTime",
    )
    act = _ts(
        pd.DataFrame({"startTime": ["2026-01-02T00:00:00Z"], "generation": [100.0]}),
        "startTime",
    )
    out = align(fc, act, "same_start")
    assert len(out) == 1
    assert out.loc[0, "error"] == 10.0  # forecast - actual: positive means over-forecast
    assert out.loc[0, "horizon_h"] == 12.0


def test_align_hour_mean_needs_both_half_hours() -> None:
    fc = _ts(
        pd.DataFrame(
            {
                "publishTime": ["2026-01-01T12:00:00Z"] * 2,
                "startTime": ["2026-01-02T00:00:00Z", "2026-01-02T01:00:00Z"],
                "forecast": [115.0, 60.0],
            }
        ),
        "publishTime",
        "startTime",
    )
    act = _ts(
        pd.DataFrame(
            {
                "startTime": [
                    "2026-01-02T00:00:00Z",
                    "2026-01-02T00:30:00Z",
                    "2026-01-02T01:00:00Z",  # 01:30 is missing, so that hour is dropped
                ],
                "generation": [100.0, 120.0, 50.0],
            }
        ),
        "startTime",
    )
    out = align(fc, act, "hour_mean")
    assert len(out) == 1
    assert out.loc[0, "actual"] == 110.0
    assert out.loc[0, "error"] == 5.0


def test_metrics_by_horizon_buckets() -> None:
    df = pd.DataFrame(
        {"forecast": [1.0, 2.0, 3.0], "actual": [1.0, 1.0, 1.0], "horizon_h": [10.0, 20.0, 20.0]}
    )
    out = metrics_by(df, "horizon")
    assert [(r["group"], r["n"]) for r in out] == [("<18h", 1), ("18-24h", 2)]


def test_metrics_by_hour_uses_uk_local_hour() -> None:
    df = _ts(
        pd.DataFrame(
            {
                  "startTime": [
                      "2026-01-02T00:00:00Z",
                      "2026-01-03T00:00:00Z",
                      "2026-01-02T05:00:00Z",
                  ],
                "forecast": [1.0, 1.0, 1.0],
                "actual": [1.0, 1.0, 1.0],
            }
        ),
        "startTime",
    )
    assert [(r["group"], r["n"]) for r in metrics_by(df, "hour")] == [("0", 2), ("5", 1)]


def test_metrics_by_wind_level_terciles() -> None:
    df = pd.DataFrame({"forecast": [1.0] * 9, "actual": [float(i) for i in range(1, 10)]})
    assert [r["group"] for r in metrics_by(df, "wind_level")] == ["low", "medium", "high"]


def test_metrics_by_rejects_unknown_group() -> None:
    with pytest.raises(ValueError):
        metrics_by(pd.DataFrame({"forecast": [1.0], "actual": [1.0]}), "nope")  # type: ignore[arg-type]


def test_add_band_uses_error_quantiles_and_clips_at_zero() -> None:
    df = pd.DataFrame({"forecast": [100.0, 10.0]})
    out = add_band(df, p10_error=-20.0, p90_error=30.0)
    assert list(out["band_low"]) == [70.0, 0.0]
    assert list(out["band_high"]) == [120.0, 30.0]