"""SQLAlchemy engine/session and schema creation."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .settings import load

settings = load()

# SQLite needs check_same_thread=False when used across FastAPI threads.
_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, future=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create all tables. Imports models so they register on Base.metadata."""
    from . import models  # noqa: F401

    Base.metadata.create_all(engine)
