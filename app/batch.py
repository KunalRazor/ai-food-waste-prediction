"""Daily batch: forecast every item, store predictions, and store the derived
order/spoilage recommendations. This is the scheduled "online path" job.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import delete, insert, select

from .db import SessionLocal, engine
from .forecast import forecast_all
from .models import Inventory, Item, Prediction, Recommendation
from .recommend import build_recommendations
from .registry import load_current


def latest_inventory() -> pd.DataFrame:
    inv = pd.read_sql(
        select(Inventory.item_id, Inventory.date, Inventory.stock_on_hand, Inventory.on_order),
        engine,
    )
    if inv.empty:
        return pd.DataFrame(columns=["item_id", "stock_on_hand", "on_order"])
    inv = inv.sort_values("date").groupby("item_id", as_index=False).last()
    return inv[["item_id", "stock_on_hand", "on_order"]]


def run_daily_batch(horizon: int | None = None) -> dict:
    _, meta = load_current()
    version = meta["version"]
    daily_rmse = float(meta.get("rmse", 0.0))

    fc = forecast_all(horizon)
    items = pd.read_sql(select(Item.id, Item.name, Item.unit_cost, Item.shelf_life_days), engine)
    recs = build_recommendations(fc, latest_inventory(), items, daily_rmse)

    today = dt.date.today()
    with SessionLocal() as s:
        s.execute(delete(Prediction))
        s.execute(insert(Prediction), [
            dict(item_id=int(r.item_id), target_date=r.target_date,
                 forecast_units=float(r.forecast_units), model_version=version)
            for r in fc.itertuples()
        ])
        s.execute(delete(Recommendation))
        s.execute(insert(Recommendation), [
            dict(item_id=r["item_id"], date=today,
                 recommended_order=r["recommended_order"],
                 spoilage_risk_qty=r["spoilage_risk_qty"],
                 money_at_risk=r["money_at_risk"])
            for r in recs
        ])
        s.commit()

    return {"model_version": version, "predictions": len(fc), "recommendations": len(recs)}


if __name__ == "__main__":
    print("daily batch:", run_daily_batch())
