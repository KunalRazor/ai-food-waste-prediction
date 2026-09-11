"""Feature engineering.

Turns raw history into a model-ready table. The golden rule is NO LEAKAGE: every
lag and rolling statistic is computed from days strictly BEFORE the target day
(we roll on `units.shift(1)`), so we never let the future leak into training.

`FEATURE_COLS` is the single source of truth for column order, shared by both
training (vectorized) and recursive forecasting (`compute_feature_row`).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sqlalchemy import select

from .db import engine
from .models import CalendarDay, Item, Sale, Weather

FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14",
    "roll_7_mean", "roll_7_std", "roll_28_mean",
    "item_mean",
    "dow", "is_weekend", "month", "is_holiday",
    "temp_c", "precip_mm", "price",
    "category_code",
]
TARGET = "units_sold"


def category_map() -> dict[str, int]:
    """Stable category -> int mapping (sorted for determinism)."""
    with engine.connect() as c:
        cats = sorted({row[0] for row in c.execute(select(Item.category))})
    return {cat: i for i, cat in enumerate(cats)}


def _raw_frame() -> pd.DataFrame:
    sales = pd.read_sql(
        select(Sale.item_id, Sale.date, Sale.units_sold, Sale.price, Sale.promo), engine
    )
    weather = pd.read_sql(select(Weather.date, Weather.temp_c, Weather.precip_mm), engine)
    cal = pd.read_sql(select(CalendarDay.date, CalendarDay.dow, CalendarDay.is_weekend,
                             CalendarDay.is_holiday), engine)
    items = pd.read_sql(select(Item.id, Item.category), engine).rename(columns={"id": "item_id"})

    df = sales.merge(weather, on="date").merge(cal, on="date").merge(items, on="item_id")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["is_holiday"] = df["is_holiday"].astype(int)
    cmap = category_map()
    df["category_code"] = df["category"].map(cmap).astype(int)
    return df.sort_values(["item_id", "date"]).reset_index(drop=True)


def build_training_frame() -> pd.DataFrame:
    """Return a DataFrame with FEATURE_COLS + target + date + item_id, warmup
    rows (without full lag history) dropped."""
    df = _raw_frame()
    parts = []
    for _, grp in df.groupby("item_id", sort=True):
        grp = grp.sort_values("date").copy()
        u = grp["units_sold"]
        past = u.shift(1)  # everything derived from `past` excludes the target day
        grp["lag_1"] = u.shift(1)
        grp["lag_7"] = u.shift(7)
        grp["lag_14"] = u.shift(14)
        grp["roll_7_mean"] = past.rolling(7).mean()
        grp["roll_7_std"] = past.rolling(7).std()
        grp["roll_28_mean"] = past.rolling(28).mean()
        grp["item_mean"] = past.expanding().mean()
        parts.append(grp)
    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["lag_14", "roll_28_mean", "roll_7_std"]).reset_index(drop=True)
    return out


def compute_feature_row(
    history: list[float],
    target_date: dt.date,
    is_holiday: bool,
    temp_c: float,
    precip_mm: float,
    price: float,
    category_code: int,
) -> dict:
    """Build one feature row for a future day from a history of past daily units
    (most recent last). Used by recursive multi-step forecasting."""
    h = np.asarray(history, dtype=float)

    def lag(n: int) -> float:
        return float(h[-n]) if len(h) >= n else float(h.mean() if len(h) else 0.0)

    def roll(n: int, fn) -> float:
        window = h[-n:] if len(h) >= 1 else np.array([0.0])
        return float(fn(window)) if len(window) else 0.0

    return {
        "lag_1": lag(1),
        "lag_7": lag(7),
        "lag_14": lag(14),
        "roll_7_mean": roll(7, np.mean),
        "roll_7_std": roll(7, lambda w: np.std(w, ddof=1) if len(w) > 1 else 0.0),
        "roll_28_mean": roll(28, np.mean),
        "item_mean": float(h.mean()) if len(h) else 0.0,
        "dow": target_date.weekday(),
        "is_weekend": int(target_date.weekday() >= 5),
        "month": target_date.month,
        "is_holiday": int(is_holiday),
        "temp_c": float(temp_c),
        "precip_mm": float(precip_mm),
        "price": float(price),
        "category_code": int(category_code),
    }
