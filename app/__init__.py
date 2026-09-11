"""AI Food Waste Prediction Platform.

Forecasts per-item daily demand for restaurants (XGBoost), then turns the
forecast into daily order recommendations and spoilage-risk flags. FastAPI
serves the results; data lives in PostgreSQL (SQLite works for a zero-setup run).
"""
