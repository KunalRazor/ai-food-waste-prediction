"""ORM models. Single-restaurant for simplicity; multi-restaurant would add a
restaurant_id to sales/inventory/predictions (see README extensions)."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Item(Base):
    __tablename__ = "items"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    category: Mapped[str] = mapped_column(String(60), index=True)
    unit_cost: Mapped[float] = mapped_column(Float)          # cost per unit (money at risk)
    base_price: Mapped[float] = mapped_column(Float)         # menu price
    shelf_life_days: Mapped[int] = mapped_column(Integer)    # how long it keeps
    perishable: Mapped[bool] = mapped_column(Boolean, default=True)

    sales: Mapped[list["Sale"]] = relationship(back_populates="item")


class Sale(Base):
    __tablename__ = "sales"
    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    units_sold: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    promo: Mapped[bool] = mapped_column(Boolean, default=False)

    item: Mapped[Item] = relationship(back_populates="sales")


class Weather(Base):
    __tablename__ = "weather"
    date: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    temp_c: Mapped[float] = mapped_column(Float)
    precip_mm: Mapped[float] = mapped_column(Float)
    humidity: Mapped[float] = mapped_column(Float)


class CalendarDay(Base):
    __tablename__ = "calendar"
    date: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    dow: Mapped[int] = mapped_column(Integer)          # 0=Mon .. 6=Sun
    is_weekend: Mapped[bool] = mapped_column(Boolean)
    is_holiday: Mapped[bool] = mapped_column(Boolean, default=False)
    holiday_name: Mapped[str] = mapped_column(String(80), default="")


class Inventory(Base):
    __tablename__ = "inventory"
    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    stock_on_hand: Mapped[float] = mapped_column(Float)
    on_order: Mapped[float] = mapped_column(Float, default=0.0)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"), index=True)
    target_date: Mapped[dt.date] = mapped_column(Date, index=True)
    forecast_units: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


class Recommendation(Base):
    __tablename__ = "recommendations"
    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    recommended_order: Mapped[float] = mapped_column(Float)
    spoilage_risk_qty: Mapped[float] = mapped_column(Float)
    money_at_risk: Mapped[float] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(64), unique=True)
    path: Mapped[str] = mapped_column(String(255))
    mae: Mapped[float] = mapped_column(Float)
    rmse: Mapped[float] = mapped_column(Float)
    mape: Mapped[float] = mapped_column(Float)
    bias: Mapped[float] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())
