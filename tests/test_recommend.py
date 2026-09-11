"""Recommendation + spoilage math (pure function, no DB needed)."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from app.recommend import build_recommendations


def _forecast(item_id: int, per_day: list[float]) -> pd.DataFrame:
    base = dt.date(2025, 1, 1)
    return pd.DataFrame([
        {"item_id": item_id, "target_date": base + dt.timedelta(days=i), "forecast_units": u}
        for i, u in enumerate(per_day)
    ])


def test_overstocked_item_flags_spoilage():
    # Shelf life 3 days, forecast 5/day -> expected use over shelf = 15.
    # Stock 40 -> 25 at risk * $2 = $50 money at risk.
    fc = _forecast(1, [5, 5, 5, 5, 5, 5, 5])
    inv = pd.DataFrame([{"item_id": 1, "stock_on_hand": 40, "on_order": 0}])
    items = pd.DataFrame([{"id": 1, "name": "Salad", "unit_cost": 2.0, "shelf_life_days": 3}])

    recs = build_recommendations(fc, inv, items, daily_rmse=0.0)
    r = recs[0]
    assert r["expected_use_over_shelf_life"] == 15
    assert r["spoilage_risk_qty"] == 25
    assert r["money_at_risk"] == 50.0
    # With daily_rmse=0 and lead=1, order = lead_demand(5) - stock(40) < 0 -> 0.
    assert r["recommended_order"] == 0


def test_understocked_item_recommends_order():
    fc = _forecast(2, [30, 30, 30])
    inv = pd.DataFrame([{"item_id": 2, "stock_on_hand": 0, "on_order": 0}])
    items = pd.DataFrame([{"id": 2, "name": "Fries", "unit_cost": 0.7, "shelf_life_days": 3}])

    recs = build_recommendations(fc, inv, items, daily_rmse=0.0)
    r = recs[0]
    # lead_time default = 1 day -> order at least the first day's demand.
    assert r["recommended_order"] >= 30
    assert r["spoilage_risk_qty"] == 0
    assert r["money_at_risk"] == 0


def test_ranked_by_money_at_risk():
    fc = pd.concat([_forecast(1, [1, 1, 1]), _forecast(2, [1, 1, 1])], ignore_index=True)
    inv = pd.DataFrame([
        {"item_id": 1, "stock_on_hand": 10, "on_order": 0},
        {"item_id": 2, "stock_on_hand": 100, "on_order": 0},
    ])
    items = pd.DataFrame([
        {"id": 1, "name": "A", "unit_cost": 1.0, "shelf_life_days": 2},
        {"id": 2, "name": "B", "unit_cost": 5.0, "shelf_life_days": 2},
    ])
    recs = build_recommendations(fc, inv, items, daily_rmse=0.0)
    assert recs[0]["item_id"] == 2  # far more money at risk -> ranked first
