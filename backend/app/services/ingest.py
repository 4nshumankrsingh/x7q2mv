"""Turn raw Elexon rows into clean, UTC-aware DataFrames and a sanity report."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _clean(
    rows: list[dict[str, Any]], cols: list[str], time_cols: list[str], keys: list[str]
) -> tuple[pd.DataFrame, int]:
    df = pd.DataFrame(rows, columns=cols)
    for c in time_cols:
        # as_unit("ns") pins the datetime resolution: pandas 3 infers it from the input,
        # and mixed resolutions make joins and comparisons fragile.
        df[c] = pd.to_datetime(df[c], utc=True).dt.as_unit("ns")
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce").astype("float64")
    before = len(df)
    df = df.dropna(subset=["generation"])
    dropped_null = before - len(df)
    df = df.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
    return df, dropped_null


def actuals_to_frame(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, int]:
    """Columns: startTime (UTC), generation (MW). Returns (frame, null rows dropped)."""
    return _clean(rows, ["startTime", "generation"], ["startTime"], ["startTime"])


def forecasts_to_frame(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, int]:
    """Columns: startTime, publishTime (UTC), generation (MW). One row per (target, publish)."""
    return _clean(
        rows,
        ["startTime", "publishTime", "generation"],
        ["startTime", "publishTime"],
        ["startTime", "publishTime"],
    )


def restrict_to_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Keep rows whose startTime is in [start, end)."""
    mask = (df["startTime"] >= start) & (df["startTime"] < end)
    return df.loc[mask].reset_index(drop=True)


def _stats(s: pd.Series) -> dict[str, float | None]:
    if s.empty:
        return {"min": None, "median": None, "max": None}
    return {"min": float(s.min()), "median": float(s.median()), "max": float(s.max())}


def build_report(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> dict[str, object]:
    expected = pd.date_range(window_start, window_end, freq="30min", inclusive="left").as_unit(
        "ns"
    )
    missing = expected.difference(pd.DatetimeIndex(actuals["startTime"]))
    # WINDFOR is published hourly, so judge forecast coverage on the hourly grid only.
    hourly = expected[expected.minute == 0]
    have = pd.DatetimeIndex(forecasts["startTime"].unique())
    covered = hourly.intersection(have)

    horizon_h = (forecasts["startTime"] - forecasts["publishTime"]) / pd.Timedelta(hours=1)
    per_target = forecasts.assign(horizon_h=horizon_h).groupby("startTime")
    versions = per_target.size()
    max_horizon = per_target["horizon_h"].max()

    has_pub = not forecasts.empty
    return {
        "expected_half_hours": len(expected),
        "actuals_rows": len(actuals),
        "missing_actual_count": len(missing),
        "missing_actual_examples": [t.isoformat() for t in missing[:10]],
        "forecast_rows": len(forecasts),
        "forecast_publish_times": int(forecasts["publishTime"].nunique()),
        "publish_first": forecasts["publishTime"].min().isoformat() if has_pub else None,
        "publish_last": forecasts["publishTime"].max().isoformat() if has_pub else None,
        "targets_with_forecast": len(covered),
        "expected_hourly_targets": len(hourly),
        "targets_without_forecast": len(hourly) - len(covered),
        "versions_per_target": _stats(versions),
        "max_horizon_h": _stats(max_horizon),
    }