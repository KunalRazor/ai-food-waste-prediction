# AI Food Waste Prediction Platform

An ML system that forecasts **how much of each item a restaurant will sell each
day**, then turns that forecast into **daily order recommendations** and flags
**items likely to spoil**. Restaurants over-order (waste) or under-order
(stockouts) because they guess demand; this learns demand from history + weather
+ the calendar.

**Stack:** Python, FastAPI, XGBoost, PostgreSQL (SQLite for zero-setup local runs).

## Two paths

- **Offline (training):** synthetic data → feature engineering → XGBoost with a
  time-based split and walk-forward backtest → saved model + metrics.
- **Online (serving):** load the model → recursive multi-step forecast → derive
  order + spoilage recommendations → FastAPI serves them.

```
 sales + weather + calendar (Postgres)
            │  build features (lags, rolling, calendar, weather) — NO leakage
            ▼
   XGBoost (Poisson objective)  ──time-based split + walk-forward backtest──▶ metrics
            │  save model + version (registry)
            ▼
   recursive per-item forecast ──▶ recommendations (order qty) + spoilage risk
            │
            ▼
        FastAPI  ( /forecast/{id}, /recommendations, /spoilage-risk )
   scheduler: daily batch + weekly retrain
```

## Components

| Path | What it is |
|---|---|
| `app/data_gen.py` | Synthetic data: weekday/seasonal/weather/promo demand (Poisson), weather, calendar, inventory |
| `app/models.py` | SQLAlchemy tables: items, sales, weather, calendar, inventory, predictions, recommendations, model_versions |
| `app/features.py` | Lag/rolling/calendar/weather features; **leakage-free** (rolls on `shift(1)`) |
| `app/train.py` | XGBoost training, **time-based split**, **walk-forward backtest**, MAE/RMSE/MAPE + bias |
| `app/registry.py` | Save/load model + metrics + version (`current.json` pointer) |
| `app/forecast.py` | Recursive multi-step forecast; future weather via day-of-year climatology |
| `app/recommend.py` | Order qty (forecast + safety stock − stock) and spoilage risk (money at risk) |
| `app/batch.py` | Daily job: forecast → store predictions + recommendations |
| `app/scheduler.py` | APScheduler: daily batch + weekly retrain (maps to K8s CronJobs) |
| `app/main.py` | FastAPI service |

## Quick start (local, zero setup — SQLite)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS: XGBoost needs OpenMP -> brew install libomp

python -m app.data_gen        # generate 2 years of synthetic data
python -m app.train           # train + backtest, prints metrics
uvicorn app.main:app --port 8000
```

Example training output (real, from synthetic data):

```
holdout: MAE=9.54 RMSE=12.93 MAPE=12.3% bias=-0.63
walk-forward backtest avg MAPE=12.1% over 4 folds
```

## API

```bash
curl localhost:8000/healthz
curl "localhost:8000/forecast/1?horizon=7"     # 7-day demand for item 1
curl localhost:8000/recommendations            # order qty per item
curl localhost:8000/spoilage-risk              # items at risk, ranked by money at risk
```

`/spoilage-risk` returns e.g. `Margherita Pizza — 100.9 units, $353 at risk`.

## Run with PostgreSQL (docker-compose)

```bash
docker compose up --build      # or docker-compose up --build
# API at http://localhost:8000/docs ; it seeds data + trains on first boot
```

## Tests

```bash
python -m pytest -q
```

Covers: **no-leakage** feature check (lag_1 equals the actual previous day, never
the target day), the recommendation/spoilage math, and API smoke tests over the
full serving path.

## How each resume bullet maps to the code

| Resume bullet | Where it lives |
|---|---|
| ML-based demand forecasting for food demand & spoilage risk | `app/train.py`, `app/forecast.py`, `app/recommend.py` |
| Trained on historical sales, weather patterns, seasonal demand | `app/data_gen.py`, `app/features.py` |
| Backend APIs for daily inventory recommendations & spoilage items | `app/main.py`, `app/batch.py` |

## Interview notes

- **No data leakage:** time-based split (train past, test recent) and lag/rolling
  features built only from days *before* the target — never a random shuffle.
- **Why XGBoost + Poisson:** tabular data, many items in one model, external
  signals (weather/holidays), and demand is a non-negative count (Poisson), not
  a bell curve. Beats ARIMA/Prophet which fit one series at a time.
- **Spoilage is derived, not a separate model:** forecast demand over the shelf
  life vs current stock; the excess is at risk, ranked by money at risk.
- **Safety stock = newsvendor:** `z × daily_error × √lead_time`; bigger model
  error → bigger buffer. Retrain weekly / on drift.

See `../Downloads/AI_Food_Waste_Prediction_Architecture.md` for the full
architecture write-up and interview Q&A.
