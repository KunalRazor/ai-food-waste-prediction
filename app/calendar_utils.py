"""Shared calendar helpers so data generation and forecasting agree on holidays."""

from __future__ import annotations

import datetime as dt

HOLIDAYS: dict[tuple[int, int], str] = {
    (1, 1): "New Year",
    (7, 4): "Independence Day",
    (10, 31): "Halloween",
    (12, 25): "Christmas",
}


def holiday_name(d: dt.date) -> str:
    return HOLIDAYS.get((d.month, d.day), "")


def is_holiday(d: dt.date) -> bool:
    return (d.month, d.day) in HOLIDAYS
