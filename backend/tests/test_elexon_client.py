from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
import respx
from tenacity import wait_none

from app.config import settings
from app.services.elexon_client import ElexonClient, iter_date_windows, iter_time_windows

FUELHH = f"{settings.elexon_base_url}/datasets/FUELHH/stream"
WINDFOR = f"{settings.elexon_base_url}/datasets/WINDFOR/stream"
T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ElexonClient._get_json.retry, "wait", wait_none())


def test_date_windows_cover_range_without_overlap() -> None:
    windows = list(iter_date_windows(date(2026, 1, 1), date(2026, 1, 31), 7))
    assert len(windows) == 5
    assert windows[0] == (date(2026, 1, 1), date(2026, 1, 7))
    assert windows[-1] == (date(2026, 1, 29), date(2026, 1, 31))
    for (_, prev_end), (nxt_start, _) in zip(windows, windows[1:], strict=False):
        assert nxt_start == prev_end + timedelta(days=1)


def test_date_windows_single_day() -> None:
    day = date(2026, 1, 1)
    assert list(iter_date_windows(day, day, 7)) == [(day, day)]


def test_time_windows_are_consecutive() -> None:
    end = T0 + timedelta(days=10)
    windows = list(iter_time_windows(T0, end, timedelta(days=7)))
    assert windows == [(T0, T0 + timedelta(days=7)), (T0 + timedelta(days=7), end)]


async def test_fetch_actuals_keeps_only_wind_and_chunks() -> None:
    first = [
        {"fuelType": "WIND", "startTime": "2026-01-01T00:00:00Z", "generation": 100},
        {"fuelType": "CCGT", "startTime": "2026-01-01T00:00:00Z", "generation": 900},
    ]
    second = [{"fuelType": "WIND", "startTime": "2026-01-08T00:00:00Z", "generation": 120}]
    with respx.mock:
        route = respx.get(FUELHH).mock(
            side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=second)]
        )
        async with ElexonClient(window_days=7) as client:
            rows = await client.fetch_actuals(date(2026, 1, 1), date(2026, 1, 10))
    assert [r["generation"] for r in rows] == [100, 120]
    assert route.call_count == 2
    params = route.calls[0].request.url.params
    assert params["settlementDateFrom"] == "2026-01-01"
    assert params["settlementDateTo"] == "2026-01-07"


async def test_unwraps_data_envelope_and_formats_publish_window() -> None:
    row = {"publishTime": "2026-01-01T06:00:00Z", "startTime": "2026-01-02T00:00:00Z"}
    with respx.mock:
        route = respx.get(WINDFOR).mock(
            return_value=httpx.Response(200, json={"data": [{**row, "generation": 5}]})
        )
        async with ElexonClient() as client:
            rows = await client.fetch_forecasts(T0, T0 + timedelta(days=1))
    assert len(rows) == 1
    assert route.calls[0].request.url.params["publishDateTimeFrom"] == "2026-01-01T00:00:00Z"


async def test_retries_on_503_then_succeeds() -> None:
    with respx.mock:
        route = respx.get(WINDFOR).mock(
            side_effect=[httpx.Response(503), httpx.Response(200, json=[])]
        )
        async with ElexonClient() as client:
            rows = await client.fetch_forecasts(T0, T0 + timedelta(days=1))
    assert rows == []
    assert route.call_count == 2


async def test_does_not_retry_on_400() -> None:
    with respx.mock:
        route = respx.get(WINDFOR).mock(return_value=httpx.Response(400, text="bad request"))
        async with ElexonClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.fetch_forecasts(T0, T0 + timedelta(days=1))
    assert route.call_count == 1