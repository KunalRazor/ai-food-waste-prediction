"""FastAPI service.

Endpoints:
  GET /healthz            -> service + current model version
  GET /forecast/{item_id} -> next-N-day demand forecast for one item
  GET /recommendations    -> per-item order + spoilage recommendations
  GET /spoilage-risk      -> items at risk of spoiling, ranked by money at risk
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from sqlalchemy import select

from .batch import latest_inventory
from .db import engine, init_db
from .forecast import forecast_all, forecast_for_item
from .models import Item
from .recommend import build_recommendations
from .registry import current_version, load_current


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Food Waste Prediction", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model_version": current_version()}


@app.get("/forecast/{item_id}")
def forecast(item_id: int, horizon: int = 7) -> dict:
    try:
        df = forecast_for_item(item_id, horizon)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if df.empty:
        raise HTTPException(status_code=404, detail=f"no forecast for item {item_id}")
    return {
        "item_id": item_id,
        "horizon": horizon,
        "forecast": [
            {"date": r.target_date.isoformat(), "units": round(float(r.forecast_units), 1)}
            for r in df.itertuples()
        ],
    }


def _recommendations() -> list[dict]:
    try:
        _, meta = load_current()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    fc = forecast_all()
    items = pd.read_sql(select(Item.id, Item.name, Item.unit_cost, Item.shelf_life_days), engine)
    return build_recommendations(fc, latest_inventory(), items, float(meta.get("rmse", 0.0)))


@app.get("/recommendations")
def recommendations() -> dict:
    recs = _recommendations()
    return {"count": len(recs), "recommendations": recs}


@app.get("/spoilage-risk")
def spoilage_risk() -> dict:
    at_risk = [r for r in _recommendations() if r["money_at_risk"] > 0]
    return {"count": len(at_risk), "items": at_risk}
