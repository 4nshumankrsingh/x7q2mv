"""Forecast-vs-actual alignment and error metrics. Error = forecast - actual."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

Alignment = Literal["same_start", "hour_mean"]
Group = Literal["horizon", "lead_time", "hour", "wind_level"]

HORIZON_BINS = [0, 18, 24, 30, 36, 48, float("inf")]
HORIZON_LABELS = ["<18h", "18-24h", "24-30h", "30-36h", "36-48h", "48h+"]
LEAD_BINS = [0, 6, 12, 18, 24, 36, 48, float("inf")]
LEAD_LABELS = ["0-6h", "6-12h", "12-18h", "18-24h", "24-36h", "36-48h", "48h+"]


def align(
      forecasts: pd.DataFrame, actuals: pd.DataFrame, how: Alignment = "same_start"
  ) -> pd.DataFrame:
    """Inner-join forecasts (startTime, publishTime, forecast) to actuals (startTime, generation).

    same_start: actual with the same startTime as the forecast.
    hour_mean: mean of the two half-hours in the forecast's hour (hours missing one are dropped).
    Adds actual, horizon_h and error (= forecast - actual).
    """
    if how == "same_start":
        act = actuals[["startTime", "generation"]].rename(columns={"generation": "actual"})
    elif how == "hour_mean":
        hourly = actuals.assign(hour=actuals["startTime"].dt.floor("h"))
        grouped = hourly.groupby("hour")["generation"].agg(["mean", "count"]).reset_index()
        grouped = grouped[grouped["count"] == 2]
        act = grouped.rename(columns={"hour": "startTime", "mean": "actual"})[
            ["startTime", "actual"]
        ]
    else:
        raise ValueError(f"unknown alignment {how!r}")
    df = forecasts.merge(act, on="startTime", how="inner")
    df["horizon_h"] = (df["startTime"] - df["publishTime"]) / pd.Timedelta(hours=1)
    df["error"] = df["forecast"] - df["actual"]
    return df.sort_values(["startTime", "publishTime"]).reset_index(drop=True)


def error_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Needs columns forecast and actual. wMAPE = sum|e| / sum(actual), safe near zero."""
    n = len(df)
    if n == 0:
        keys = ["mae", "rmse", "bias", "wmape", "p10_error", "p90_error", "p90_abs_error"]
        return {"n": 0, **dict.fromkeys(keys)}
    e = df["forecast"] - df["actual"]
    total = float(df["actual"].sum())
    return {
        "n": n,
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt((e**2).mean())),
        "bias": float(e.mean()),
        "wmape": float(e.abs().sum() / total) if total > 0 else None,
        "p10_error": float(e.quantile(0.1)),
        "p90_error": float(e.quantile(0.9)),
        "p90_abs_error": float(e.abs().quantile(0.9)),
    }


def _wind_level(actual: pd.Series) -> pd.Series:
    try:
        return pd.qcut(actual, 3, labels=["low", "medium", "high"])
    except ValueError:  # too few distinct values for terciles
        return pd.Series("all", index=actual.index)


def metrics_by(df: pd.DataFrame, group: Group, tz: str = "Europe/London") -> list[dict[str, Any]]:
    """Metrics per group. wind_level uses terciles of actual within the given sample."""
    if df.empty:
        return []
    if group == "horizon":
        key = pd.cut(df["horizon_h"], HORIZON_BINS, labels=HORIZON_LABELS, right=False)
    elif group == "lead_time":
        key = pd.cut(df["horizon_h"], LEAD_BINS, labels=LEAD_LABELS, right=False)
    elif group == "hour":
        key = df["startTime"].dt.tz_convert(tz).dt.hour
    elif group == "wind_level":
        key = _wind_level(df["actual"])
    else:
        raise ValueError(f"unknown group {group!r}")
    return [
        {"group": str(label), **error_metrics(part)}
        for label, part in df.groupby(key, observed=True)
    ]


def add_band(df: pd.DataFrame, p10_error: float, p90_error: float) -> pd.DataFrame:
    """Plausible range for the actual: since error = forecast - actual, a high error means
    a low actual, so band_low uses P90 and band_high uses P10. Clipped at zero MW."""
    out = df.copy()
    out["band_low"] = (out["forecast"] - p90_error).clip(lower=0.0)
    out["band_high"] = out["forecast"] - p10_error
    return out