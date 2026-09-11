"""Synthetic data generator.

Produces realistic restaurant data so a real model can be trained with no
external dataset: daily per-item demand driven by weekday, season, weather, and
promotions (Poisson noise), plus weather, a calendar with holidays, and a
current-day inventory snapshot (some items overstocked to create spoilage risk).
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
from sqlalchemy import delete, insert

from .calendar_utils import HOLIDAYS
from .db import SessionLocal, init_db
from .models import (
    CalendarDay,
    Inventory,
    Item,
    Sale,
    Weather,
)

# name, category, base_demand, temp_sensitivity, unit_cost, base_price, shelf_life_days
_ITEMS = [
    ("Iced Coffee", "beverage", 60, 0.020, 0.8, 3.5, 2),
    ("Hot Coffee", "beverage", 80, -0.015, 0.5, 3.0, 1),
    ("Lemonade", "beverage", 55, 0.025, 0.6, 3.0, 3),
    ("Hot Chocolate", "beverage", 35, -0.030, 0.7, 3.5, 1),
    ("Green Salad", "salad", 40, 0.015, 2.0, 8.0, 2),
    ("Caesar Salad", "salad", 38, 0.012, 2.4, 9.5, 2),
    ("Tomato Soup", "soup", 30, -0.030, 1.5, 6.0, 3),
    ("Cheeseburger", "main", 90, 0.000, 3.0, 10.0, 2),
    ("Veggie Wrap", "main", 45, 0.005, 2.2, 9.0, 2),
    ("Margherita Pizza", "main", 70, 0.000, 3.5, 12.0, 2),
    ("Chicken Wings", "main", 65, 0.000, 2.8, 11.0, 2),
    ("French Fries", "side", 120, 0.000, 0.7, 4.0, 3),
    ("Ice Cream", "dessert", 50, 0.030, 1.0, 5.0, 30),
    ("Fruit Bowl", "dessert", 30, 0.020, 1.8, 6.0, 2),
    ("Bagel", "bakery", 75, -0.005, 0.5, 2.5, 1),
]

# Restaurants are busier toward the weekend.
_WEEKDAY_FACTOR = [0.90, 0.90, 1.00, 1.05, 1.20, 1.40, 1.30]  # Mon..Sun


def _seasonal_temp(day_of_year: int) -> float:
    # Northern-hemisphere sinusoid: coldest ~Jan, warmest ~Jul.
    return 15.0 + 12.0 * math.sin(2 * math.pi * (day_of_year - 105) / 365.0)


def generate(years: float = 2.0, seed: int = 42, end: dt.date | None = None) -> dict:
    """Generate and load all data. Returns a small summary dict."""
    rng = np.random.default_rng(seed)
    end = end or dt.date.today()
    start = end - dt.timedelta(days=int(years * 365))
    dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]

    init_db()
    with SessionLocal() as s:
        # Idempotent: clear existing rows so re-running is clean.
        for model in (Sale, Weather, CalendarDay, Inventory, Item):
            s.execute(delete(model))
        s.commit()

        # Items
        item_rows = [
            dict(id=i + 1, name=n, category=c, unit_cost=uc, base_price=bp,
                 shelf_life_days=sl, perishable=True)
            for i, (n, c, bd, ts, uc, bp, sl) in enumerate(_ITEMS)
        ]
        s.execute(insert(Item), item_rows)

        # Weather + calendar
        weather_rows, cal_rows = [], []
        temp_by_date: dict[dt.date, float] = {}
        for d in dates:
            doy = d.timetuple().tm_yday
            temp = _seasonal_temp(doy) + rng.normal(0, 2.5)
            precip = max(0.0, rng.gamma(1.2, 2.0) - 2.0)
            temp_by_date[d] = temp
            weather_rows.append(dict(date=d, temp_c=round(temp, 1),
                                     precip_mm=round(precip, 1), humidity=round(rng.uniform(40, 90), 0)))
            hol = HOLIDAYS.get((d.month, d.day), "")
            cal_rows.append(dict(date=d, dow=d.weekday(), is_weekend=d.weekday() >= 5,
                                 is_holiday=bool(hol), holiday_name=hol))
        s.execute(insert(Weather), weather_rows)
        s.execute(insert(CalendarDay), cal_rows)

        # Sales
        sale_rows = []
        recent_mean: dict[int, float] = {}
        for idx, (name, cat, base, tsens, uc, bp, sl) in enumerate(_ITEMS):
            item_id = idx + 1
            trend = rng.uniform(-0.0002, 0.0004)  # slow drift per day
            recent = []
            for day_i, d in enumerate(dates):
                temp = temp_by_date[d]
                weekday = _WEEKDAY_FACTOR[d.weekday()]
                season = 1.0 + 0.10 * math.sin(2 * math.pi * (d.timetuple().tm_yday - 105) / 365.0)
                weather = math.exp(tsens * (temp - 15.0))
                holiday = 1.25 if HOLIDAYS.get((d.month, d.day)) else 1.0
                promo = rng.random() < 0.06
                promo_boost = 1.5 if promo else 1.0
                mu = base * weekday * season * weather * holiday * promo_boost * (1 + trend * day_i)
                units = int(rng.poisson(max(0.1, mu)))
                price = round(bp * (0.9 if promo else 1.0), 2)
                sale_rows.append(dict(item_id=item_id, date=d, units_sold=units,
                                      price=price, promo=promo))
                if day_i >= len(dates) - 7:
                    recent.append(units)
            recent_mean[item_id] = float(np.mean(recent)) if recent else base
        s.execute(insert(Sale), sale_rows)

        # Current inventory snapshot (as of the last date). Some items are
        # deliberately overstocked (factor > shelf life of demand) to create
        # spoilage risk; others are short so they need ordering.
        inv_rows = []
        for idx, (name, cat, base, tsens, uc, bp, sl) in enumerate(_ITEMS):
            item_id = idx + 1
            factor = rng.uniform(0.4, 2.2)
            stock = round(recent_mean[item_id] * min(sl, 5) * factor, 1)
            inv_rows.append(dict(item_id=item_id, date=end, stock_on_hand=stock, on_order=0.0))
        s.execute(insert(Inventory), inv_rows)

        s.commit()
        return {
            "items": len(_ITEMS),
            "days": len(dates),
            "sales_rows": len(sale_rows),
            "start": start.isoformat(),
            "end": end.isoformat(),
        }


if __name__ == "__main__":
    summary = generate()
    print("generated:", summary)
