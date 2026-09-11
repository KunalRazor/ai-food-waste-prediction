"""Test setup: isolate the DB and model dir to temp locations BEFORE any app
module is imported, then seed a small dataset and train one model for the
session. Run with:  python -m pytest
"""

from __future__ import annotations

import os
import tempfile

# Must be set before importing app.* (settings are read at import time).
_tmp = tempfile.mkdtemp(prefix="foodwaste_test_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/test.db")
os.environ.setdefault("MODEL_DIR", f"{_tmp}/models")

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def seed_data_and_model():
    from app.data_gen import generate
    from app.train import train_and_register

    generate(years=0.6, seed=1)   # ~220 days x 15 items, quick to train
    train_and_register()
    yield
