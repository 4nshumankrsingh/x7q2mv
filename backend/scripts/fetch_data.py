"""Fetch wind actuals (FUELHH) and forecasts (WINDFOR) from Elexon for a target window.

Run from backend/:  python -m scripts.fetch_data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import ROOT, settings
from app.services.elexon_client import ElexonClient
from app.services.ingest import (
    actuals_to_frame,
    build_report,
    forecasts_to_frame,
    restrict_to_window,
)

RAW_DIR = ROOT / "data" / "raw"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window-start", default="2026-01-01", help="first target day, inclusive")
    p.add_argument("--window-end", default="2026-02-01", help="end of target window, exclusive")
    p.add_argument(
        "--publish-buffer-days",
        type=int,
        default=3,
        help="also fetch forecasts published this many days before window-start",
    )
    p.add_argument("--chunk-days", type=int, default=7, help="days per API request")
    return p.parse_args(argv)


def _dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    w_start = pd.Timestamp(args.window_start, tz="UTC")
    w_end = pd.Timestamp(args.window_end, tz="UTC")
    pub_start = w_start - pd.Timedelta(days=args.publish_buffer_days)

    async with ElexonClient(window_days=args.chunk_days) as client:
        last_day = (w_end - pd.Timedelta(days=1)).date()
        actual_rows = await client.fetch_actuals(w_start.date(), last_day)
        forecast_rows = await client.fetch_forecasts(
            pub_start.to_pydatetime(), w_end.to_pydatetime()
        )

    _dump(RAW_DIR / "actuals_wind_raw.json", actual_rows)
    _dump(RAW_DIR / "forecasts_raw.json", forecast_rows)

    actuals, actuals_dropped = actuals_to_frame(actual_rows)
    forecasts, forecasts_dropped = forecasts_to_frame(forecast_rows)
    actuals = restrict_to_window(actuals, w_start, w_end)
    forecasts = restrict_to_window(forecasts, w_start, w_end)  # keeps targets in the window

    if actuals.empty or forecasts.empty:
        print("ERROR: no data returned for the requested window", file=sys.stderr)
        return 1

    out = settings.data_dir
    out.mkdir(parents=True, exist_ok=True)
    actuals.to_parquet(out / "actuals.parquet", index=False)
    forecasts.to_parquet(out / "forecasts.parquet", index=False)

    meta = {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window_start": w_start.isoformat(),
        "window_end": w_end.isoformat(),
        "publish_buffer_days": args.publish_buffer_days,
        "dropped_null_actuals": actuals_dropped,
        "dropped_null_forecasts": forecasts_dropped,
        **build_report(actuals, forecasts, w_start, w_end),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(asyncio.run(run(parse_args())))