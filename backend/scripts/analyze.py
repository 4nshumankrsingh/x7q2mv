"""Run selection and metrics on the committed snapshot: python -m scripts.analyze."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.config import settings
from app.services.metrics import align, error_metrics, metrics_by
from app.services.selection import select_decision_forecasts, share_changed


def _load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(settings.data_dir / name)
    for col in ("startTime", "publishTime"):
        if col in df:
            df[col] = df[col].dt.as_unit("ns")
    return df


def main() -> None:
    actuals = _load("actuals.parquet")
    fc = _load("forecasts.parquet")
    base = settings.default_cutoff
    sel = select_decision_forecasts(fc, base)
    horizon = sel["horizon_h"]

    same = align(sel, actuals, "same_start")
    mean = align(sel, actuals, "hour_mean")
    every = align(fc.rename(columns={"generation": "forecast"}), actuals, "same_start")

    out: dict[str, Any] = {
        "cutoff": base,
        "publish_times_of_day": fc["publishTime"]
        .drop_duplicates()
        .dt.strftime("%H:%M")
        .value_counts()
        .sort_index()
        .to_dict(),
        "hourly_targets": int(fc["startTime"].nunique()),
        "selected_targets": len(sel),
        "selected_horizon_h": {
            "min": float(horizon.min()),
            "median": float(horizon.median()),
            "max": float(horizon.max()),
        },
        "share_selection_changed_vs": {
            c: round(share_changed(fc, base, c), 3) for c in ["09:00", "09:50", "12:00"]
        },
        "overall_same_start": error_metrics(same),
        "overall_hour_mean": error_metrics(mean),
        "by_horizon": metrics_by(same, "horizon"),
        "by_hour_mae": {r["group"]: round(r["mae"], 1) for r in metrics_by(same, "hour")},
        "by_wind_level": metrics_by(same, "wind_level"),
        "lead_time_all_versions": [
            {k: r[k] for k in ("group", "n", "mae", "bias")} for r in metrics_by(every, "lead_time")
        ],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()