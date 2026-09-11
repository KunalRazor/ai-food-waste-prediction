"""Feature tests, with the critical no-leakage check."""

from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import select

from app.db import engine
from app.features import FEATURE_COLS, build_training_frame, compute_feature_row
from app.models import Sale


def test_compute_feature_row_uses_only_past():
    history = [10, 12, 8, 15, 9, 11, 20]  # most recent last
    row = compute_feature_row(
        history=history, target_date=dt.date(2025, 6, 15), is_holiday=False,
        temp_c=20.0, precip_mm=0.0, price=3.5, category_code=1,
    )
    assert row["lag_1"] == 20            # yesterday
    assert row["lag_7"] == 10            # 7 days ago
    assert abs(row["roll_7_mean"] - (sum(history) / 7)) < 1e-9
    # the row never receives the target day's value, so it cannot leak.
    assert set(FEATURE_COLS) == set(row.keys())


def test_training_frame_has_no_leakage():
    df = build_training_frame()
    assert not df.empty
    assert df[FEATURE_COLS].isna().sum().sum() == 0  # warmup rows dropped

    # For a sampled row, lag_1 must equal the item's ACTUAL previous-day units,
    # never the target day's units (which would be leakage).
    raw = pd.read_sql(select(Sale.item_id, Sale.date, Sale.units_sold), engine)
    raw["date"] = pd.to_datetime(raw["date"])
    sample = df.sort_values(["item_id", "date"]).iloc[len(df) // 2]
    item_id = int(sample["item_id"])
    target_date = pd.to_datetime(sample["date"])
    prev = raw[(raw["item_id"] == item_id) & (raw["date"] == target_date - pd.Timedelta(days=1))]
    assert not prev.empty
    assert float(sample["lag_1"]) == float(prev["units_sold"].iloc[0])
