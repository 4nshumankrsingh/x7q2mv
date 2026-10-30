"""Decision-time forecast selection: use only information available at the cutoff."""

from __future__ import annotations

import re

import pandas as pd

LONDON = "Europe/London"
_HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_COLS = ["startTime", "publishTime", "generation", "cutoff"]


def parse_cutoff(hhmm: str) -> pd.Timedelta:
    """Parse a strict 24h 'HH:MM' string into a time-of-day offset."""
    m = _HHMM.match(hhmm)
    if not m:
        raise ValueError(f"cutoff must be 'HH:MM' (24h), got {hhmm!r}")
    return pd.Timedelta(hours=int(m[1]), minutes=int(m[2]))


def decision_cutoff(start: pd.Series, hhmm: str) -> pd.Series:
    """UTC cutoff for each target: HH:MM UK local time on the day before its delivery day.

    The delivery day is the UK local date of the target. The cutoff is built in naive local
    time and localised afterwards, so clock changes cannot shift it by an hour.
    """
    offset = parse_cutoff(hhmm)
    local_day = start.dt.tz_convert(LONDON).dt.tz_localize(None).dt.normalize()
    local_cutoff = local_day - pd.Timedelta(days=1) + offset
    return local_cutoff.dt.tz_localize(LONDON).dt.tz_convert("UTC")


def select_decision_forecasts(forecasts: pd.DataFrame, hhmm: str) -> pd.DataFrame:
    """For each target, the latest forecast published at or before its cutoff.

    Input: startTime, publishTime, generation. Output: startTime, publishTime, forecast,
    cutoff, horizon_h (hours between publication and target). Targets without an eligible
    forecast are dropped.
    """
    fc = forecasts.reset_index(drop=True)
    withcut = fc.assign(cutoff=decision_cutoff(fc["startTime"], hhmm))
    eligible = withcut.loc[withcut["publishTime"] <= withcut["cutoff"]]
    if eligible.empty:
        out = eligible[_COLS]
    else:
        idx = eligible.groupby("startTime")["publishTime"].idxmax()
        out = eligible.loc[idx, _COLS]
    out = out.rename(columns={"generation": "forecast"}).copy()
    out["horizon_h"] = (out["startTime"] - out["publishTime"]) / pd.Timedelta(hours=1)
    return out.sort_values("startTime").reset_index(drop=True)


def share_changed(forecasts: pd.DataFrame, base: str, other: str) -> float:
    """Share of targets whose selected forecast differs between two cutoffs."""
    a = select_decision_forecasts(forecasts, base).set_index("startTime")["publishTime"]
    b = select_decision_forecasts(forecasts, other).set_index("startTime")["publishTime"]
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1)
    return float((joined["a"] != joined["b"]).mean())