import pandas as pd
import pytest

from app.services.selection import (
    decision_cutoff,
    parse_cutoff,
    select_decision_forecasts,
    share_changed,
)

TARGET = "2026-01-15T12:00:00Z"  # delivery day 15 Jan, so the cutoff is 14 Jan 09:20 UK


def _fc(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["publishTime", "startTime", "generation"])
    for col in ("publishTime", "startTime"):
        df[col] = pd.to_datetime(df[col], utc=True).dt.as_unit("ns")
    return df


def _cutoff_of(target: str, hhmm: str = "09:20") -> pd.Timestamp:
    s = pd.Series(pd.to_datetime([target], utc=True)).dt.as_unit("ns")
    return decision_cutoff(s, hhmm).iloc[0]


def test_picks_latest_forecast_at_or_before_cutoff() -> None:
    fc = _fc(
        [
            ("2026-01-14T06:00:00Z", TARGET, 100),
            ("2026-01-14T08:30:00Z", TARGET, 110),
            ("2026-01-14T11:30:00Z", TARGET, 120),  # after the cutoff: must be ignored
        ]
    )
    sel = select_decision_forecasts(fc, "09:20")
    assert list(sel["forecast"]) == [110]
    assert (sel["publishTime"] <= sel["cutoff"]).all()  # no look-ahead


def test_forecast_published_exactly_at_cutoff_is_eligible() -> None:
    fc = _fc([("2026-01-14T08:30:00Z", TARGET, 110), ("2026-01-14T09:20:00Z", TARGET, 130)])
    assert list(select_decision_forecasts(fc, "09:20")["forecast"]) == [130]


def test_target_without_eligible_forecast_is_dropped() -> None:
    fc = _fc([("2026-01-14T11:30:00Z", TARGET, 120)])
    sel = select_decision_forecasts(fc, "09:20")
    assert len(sel) == 0
    assert "forecast" in sel.columns


def test_each_target_is_selected_independently() -> None:
    t2 = "2026-01-15T13:00:00Z"
    fc = _fc(
        [
            ("2026-01-14T08:30:00Z", TARGET, 1),
            ("2026-01-14T06:00:00Z", t2, 2),
            ("2026-01-14T08:30:00Z", t2, 3),
        ]
    )
    assert list(select_decision_forecasts(fc, "09:20")["forecast"]) == [1, 3]


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("2026-01-15T12:00:00Z", "2026-01-14T09:20:00Z"),  # winter: UK time == UTC
        ("2026-07-15T12:00:00Z", "2026-07-14T08:20:00Z"),  # summer: 09:20 BST == 08:20 UTC
        ("2026-07-14T23:30:00Z", "2026-07-14T08:20:00Z"),  # 00:30 BST on 15 Jul is the 15th
        ("2026-03-30T12:00:00Z", "2026-03-29T08:20:00Z"),  # D-1 is the clock-change day
        ("2026-03-29T00:30:00Z", "2026-03-28T09:20:00Z"),  # just before clocks go forward
    ],
)
def test_cutoff_is_uk_local_time_and_dst_safe(target: str, expected: str) -> None:
    assert _cutoff_of(target) == pd.Timestamp(expected)


def test_parse_cutoff_accepts_hhmm() -> None:
    assert parse_cutoff("09:20") == pd.Timedelta(hours=9, minutes=20)


@pytest.mark.parametrize("bad", ["9:20", "24:00", "09:60", "0920", "abc", ""])
def test_parse_cutoff_rejects_bad_input(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_cutoff(bad)


def test_share_changed_reflects_cutoff_sensitivity() -> None:
    fc = _fc([("2026-01-14T08:30:00Z", TARGET, 1), ("2026-01-14T09:30:00Z", TARGET, 2)])
    assert share_changed(fc, "09:20", "09:50") == 1.0  # 09:30 publication falls in between
    assert share_changed(fc, "09:00", "09:20") == 0.0