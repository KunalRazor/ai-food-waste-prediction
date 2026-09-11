"""Order recommendations and spoilage-risk ranking.

Both are DERIVED from the demand forecast (there is no separate spoilage model):

  order      = forecast over lead time + safety stock - (stock on hand + on order)
  safety     = z * daily_forecast_error * sqrt(lead_time)      (newsvendor buffer)
  spoilage   = stock on hand - forecast demand over the item's shelf life
  money_risk = spoilage * unit_cost                            (rank by this)
"""

from __future__ import annotations

import math

import pandas as pd

from .settings import load

settings = load()


def build_recommendations(
    forecast_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    items_df: pd.DataFrame,
    daily_rmse: float,
) -> list[dict]:
    """Combine forecast + current inventory + item metadata into per-item
    order and spoilage-risk recommendations.

    forecast_df : columns item_id, target_date, forecast_units (sorted ascending)
    inventory_df: columns item_id, stock_on_hand, on_order
    items_df    : columns id, name, unit_cost, shelf_life_days
    daily_rmse  : model's daily forecast error (sizes the safety stock)
    """
    lead = settings.lead_time_days
    z = settings.service_level_z
    safety = z * daily_rmse * math.sqrt(max(1, lead))

    inv = inventory_df.set_index("item_id")
    fc = forecast_df.sort_values("target_date")

    out: list[dict] = []
    for _, item in items_df.iterrows():
        item_id = int(item["id"])
        item_fc = fc[fc["item_id"] == item_id]["forecast_units"].to_numpy()

        # Cold start: no forecast -> fall back to a flat category-free estimate.
        if item_fc.size == 0:
            continue

        stock_on_hand = float(inv.loc[item_id, "stock_on_hand"]) if item_id in inv.index else 0.0
        on_order = float(inv.loc[item_id, "on_order"]) if item_id in inv.index else 0.0

        lead_demand = float(item_fc[:lead].sum())
        recommended = max(0.0, lead_demand + safety - (stock_on_hand + on_order))

        shelf = int(item["shelf_life_days"])
        expected_use = float(item_fc[:shelf].sum())
        spoilage_qty = max(0.0, stock_on_hand - expected_use)
        money_at_risk = spoilage_qty * float(item["unit_cost"])

        out.append({
            "item_id": item_id,
            "name": item["name"],
            "stock_on_hand": round(stock_on_hand, 1),
            "lead_time_demand": round(lead_demand, 1),
            "safety_stock": round(safety, 1),
            "recommended_order": round(recommended, 1),
            "shelf_life_days": shelf,
            "expected_use_over_shelf_life": round(expected_use, 1),
            "spoilage_risk_qty": round(spoilage_qty, 1),
            "money_at_risk": round(money_at_risk, 2),
        })

    out.sort(key=lambda r: r["money_at_risk"], reverse=True)
    return out
