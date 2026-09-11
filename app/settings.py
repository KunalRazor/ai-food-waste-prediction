"""Runtime settings.

Defaults to a local SQLite file so the whole pipeline runs with zero setup.
Set DATABASE_URL (e.g. the docker-compose Postgres) to use PostgreSQL, which is
the production/resume database.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    model_dir: str
    # Business knobs (see recommend.py):
    lead_time_days: int       # days between ordering and delivery
    service_level_z: float    # safety-stock z-score (0.95 service ≈ 1.65)
    forecast_horizon: int     # days to forecast ahead


def load() -> Settings:
    return Settings(
        database_url=os.getenv("DATABASE_URL", "sqlite:///./foodwaste.db"),
        model_dir=os.getenv("MODEL_DIR", "./models"),
        lead_time_days=int(os.getenv("LEAD_TIME_DAYS", "1")),
        service_level_z=float(os.getenv("SERVICE_LEVEL_Z", "1.65")),
        forecast_horizon=int(os.getenv("FORECAST_HORIZON", "7")),
    )
