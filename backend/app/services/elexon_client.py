"""Async client for the Elexon Insights stream endpoints."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any, Self

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRY_STATUS
    return isinstance(exc, httpx.TransportError)


def iter_date_windows(start: date, end: date, days: int) -> Iterator[tuple[date, date]]:
    """Yield inclusive, non-overlapping (from, to) date windows covering start..end."""
    if days < 1:
        raise ValueError("days must be >= 1")
    cur = start
    while cur <= end:
        last = min(cur + timedelta(days=days - 1), end)
        yield cur, last
        cur = last + timedelta(days=1)


def iter_time_windows(
    start: datetime, end: datetime, step: timedelta
) -> Iterator[tuple[datetime, datetime]]:
    """Yield consecutive (from, to) windows covering start..end."""
    cur = start
    while cur < end:
        nxt = min(cur + step, end)
        yield cur, nxt
        cur = nxt


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class ElexonClient:
    def __init__(self, *, window_days: int = 7, http: httpx.AsyncClient | None = None) -> None:
        self.window_days = window_days
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(base_url=settings.elexon_base_url, timeout=60.0)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._owns_http:
            await self._http.aclose()

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=20),
        reraise=True,
    )
    async def _get_json(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        resp = await self._http.get(path, params=params)
        if resp.is_error:
            log.warning("HTTP %s from %s: %s", resp.status_code, resp.url, resp.text[:300])
        resp.raise_for_status()
        payload = resp.json()
        rows = payload["data"] if isinstance(payload, dict) else payload
        return list(rows)

    async def fetch_actuals(self, start: date, end: date) -> list[dict[str, Any]]:
        """Wind rows from FUELHH for settlement dates start..end (inclusive)."""
        rows: list[dict[str, Any]] = []
        for d0, d1 in iter_date_windows(start, end, self.window_days):
            params = {"settlementDateFrom": d0.isoformat(), "settlementDateTo": d1.isoformat()}
            chunk = await self._get_json("/datasets/FUELHH/stream", params)
            wind = [r for r in chunk if r.get("fuelType") == "WIND"]
            log.info("FUELHH %s..%s: %d rows, %d wind", d0, d1, len(chunk), len(wind))
            rows.extend(wind)
        return rows

    async def fetch_forecasts(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """WINDFOR rows published between start and end."""
        rows: list[dict[str, Any]] = []
        step = timedelta(days=self.window_days)
        for w0, w1 in iter_time_windows(start, end, step):
            params = {"publishDateTimeFrom": _iso(w0), "publishDateTimeTo": _iso(w1)}
            chunk = await self._get_json("/datasets/WINDFOR/stream", params)
            log.info("WINDFOR published %s..%s: %d rows", _iso(w0), _iso(w1), len(chunk))
            rows.extend(chunk)
        return rows