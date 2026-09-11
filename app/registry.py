"""Model registry (lightweight).

Saves the trained model + its metrics + a version to disk, records a row in the
`model_versions` table, and maintains a `current.json` pointer the serving code
loads. A production system would use MLflow; this keeps the same idea small.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import xgboost as xgb

from .db import SessionLocal
from .models import ModelVersion
from .settings import load

settings = load()


def _paths(version: str) -> tuple[str, str]:
    model_path = os.path.join(settings.model_dir, f"model_{version}.json")
    meta_path = os.path.join(settings.model_dir, f"meta_{version}.json")
    return model_path, meta_path


def save_model(model: xgb.XGBRegressor, metrics: dict) -> str:
    os.makedirs(settings.model_dir, exist_ok=True)
    version = dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S")
    model_path, meta_path = _paths(version)

    model.save_model(model_path)
    meta = {"version": version, "path": model_path, **metrics}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(settings.model_dir, "current.json"), "w") as f:
        json.dump({"version": version}, f)

    with SessionLocal() as s:
        s.add(ModelVersion(
            version=version, path=model_path,
            mae=metrics.get("mae", 0.0), rmse=metrics.get("rmse", 0.0),
            mape=metrics.get("mape", 0.0), bias=metrics.get("bias", 0.0),
        ))
        s.commit()
    return version


def current_version() -> str | None:
    ptr = os.path.join(settings.model_dir, "current.json")
    if not os.path.exists(ptr):
        return None
    with open(ptr) as f:
        return json.load(f).get("version")


def load_current() -> tuple[xgb.XGBRegressor, dict]:
    version = current_version()
    if version is None:
        raise FileNotFoundError("no current model; run training first (python -m app.train)")
    model_path, meta_path = _paths(version)
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta
