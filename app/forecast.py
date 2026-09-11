"""Recursive multi-step forecasting.

Loads the current model and forecasts the next H days per item. Each predicted
day is fed back into the history so the next day's lag/rolling features are
consistent (true multi-step forecasting). Future weather is approximated by
day-of-year climatology from the weather table (a stand-in for a weather API).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sqlalchemy import select

from .calendar_utils import is_holiday
from .db import engine
from .features import FEATURE_COLS, category_map, compute_feature_row
from .models import Item, Sale, Weather
from .registry import load_current
from .settings import load

settings = load()
_HISTORY_DAYS = 40  # enough to fill lag_14 + roll_28


def _climatology() -> pd.DataFrame:
    w = pd.read_sql(select(Weather.date, Weather.temp_c, Weather.precip_mm), engine)
    w["doy"] = pd.to_datetime(w["date"]).dt.dayofyear
    return w.groupby("doy")[["temp_c", "precip_mm"]].mean()


def _weather_for(clim: pd.DataFrame, d: dt.date) -> tuple[float, float]:
    doy = d.timetuple().tm_yday
    if doy in clim.index:
        row = clim.loc[doy]
        return float(row["temp_c"]), float(row["precip_mm"])
    return float(clim["temp_c"].mean()), float(clim["precip_mm"].mean())


def _histories() -> tuple[dict[int, list[float]], dt.date]:
    sales = pd.read_sql(select(Sale.item_id, Sale.date, Sale.units_sold), engine)
    sales["date"] = pd.to_datetime(sales["date"])
    last_date = sales["date"].max().date()
    hist: dict[int, list[float]] = {}
    for item_id, grp in sales.sort_values("date").groupby("item_id"):
        hist[int(item_id)] = grp["units_sold"].tolist()[-_HISTORY_DAYS:]
    return hist, last_date


def forecast_all(horizon: int | None = None) -> pd.DataFrame:
    """Return a DataFrame [item_id, target_date, forecast_units] for the next
    `horizon` days for every item."""
    horizon = horizon or settings.forecast_horizon
    model, meta = load_current()
    cmap = category_map()
    clim = _climatology()
    items = pd.read_sql(select(Item.id, Item.category, Item.base_price), engine)
    hist, last_date = _histories()

    rows = []
    for _, item in items.iterrows():
        item_id = int(item["id"])
        h = list(hist.get(item_id, []))
        code = cmap.get(item["category"], 0)
        for step in range(1, horizon + 1):
            target = last_date + dt.timedelta(days=step)
            temp, precip = _weather_for(clim, target)
            feat = compute_feature_row(
                history=h, target_date=target, is_holiday=is_holiday(target),
                temp_c=temp, precip_mm=precip, price=float(item["base_price"]),
                category_code=code,
            )
            X = pd.DataFrame([feat])[FEATURE_COLS]
            pred = float(np.clip(model.predict(X)[0], 0.0, None))
            rows.append({"item_id": item_id, "target_date": target, "forecast_units": pred})
            h.append(pred)  # feed prediction back for the next step
    return pd.DataFrame(rows)


def forecast_for_item(item_id: int, horizon: int | None = None) -> pd.DataFrame:
    df = forecast_all(horizon)
    return df[df["item_id"] == item_id].reset_index(drop=True)
