"""Train the demand model.

Frames demand as supervised regression (units sold per item per day) and fits
XGBoost with a Poisson objective (demand is a non-negative count, not a bell
curve). Evaluation respects time: a time-based holdout plus a walk-forward
backtest — never a random shuffle, which would leak the future into the past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import FEATURE_COLS, TARGET, build_training_frame
from .registry import save_model


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    denom = np.clip(np.abs(y_true), 1.0, None)  # avoid /0 on zero-demand days
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err) / denom) * 100.0),
        "bias": float(np.mean(err)),  # +ve = over-forecasting
    }


def _fit(df: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        n_jobs=4,
        random_state=42,
    )
    model.fit(df[FEATURE_COLS], df[TARGET])
    return model


def _predict(model: xgb.XGBRegressor, df: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(df[FEATURE_COLS]), 0.0, None)


def time_split(df: pd.DataFrame, frac: float = 0.8):
    """Train on the earliest `frac` of DATES, validate on the most recent."""
    dates = np.sort(df["date"].unique())
    cutoff = dates[int(len(dates) * frac)]
    return df[df["date"] < cutoff], df[df["date"] >= cutoff], cutoff


def walk_forward(df: pd.DataFrame, folds: int = 4, horizon_frac: float = 0.05) -> list[dict]:
    """Expanding-window backtest: train up to a cutoff, validate the next slice,
    roll the cutoff forward. Mimics how the model is used in production."""
    dates = np.sort(df["date"].unique())
    n = len(dates)
    step = max(1, int(n * horizon_frac))
    start = max(step, n - folds * step)
    out = []
    for i in range(folds):
        cut = start + i * step
        if cut >= n:
            break
        end = min(cut + step, n)
        tr = df[df["date"] < dates[cut]]
        if end < n:
            va = df[(df["date"] >= dates[cut]) & (df["date"] < dates[end])]
        else:
            va = df[df["date"] >= dates[cut]]  # last fold runs to the end
        if len(tr) == 0 or len(va) == 0:
            continue
        m = _fit(tr)
        out.append(metrics(va[TARGET].to_numpy(), _predict(m, va)))
    return out


def train_and_register() -> tuple[str, dict, list[dict]]:
    df = build_training_frame()
    if df.empty:
        raise RuntimeError("no training data; run `python -m app.data_gen` first")

    # Honest holdout estimate.
    train_df, valid_df, _ = time_split(df, 0.8)
    holdout_model = _fit(train_df)
    m = metrics(valid_df[TARGET].to_numpy(), _predict(holdout_model, valid_df))

    # Walk-forward backtest for reporting.
    bt = walk_forward(df)
    if bt:
        m["backtest_mape"] = float(np.mean([f["mape"] for f in bt]))

    # Production model: refit on ALL data (uses the most recent days too).
    final_model = _fit(df)
    version = save_model(final_model, m)
    return version, m, bt


if __name__ == "__main__":
    version, m, bt = train_and_register()
    print(f"trained model version={version}")
    print(f"holdout: MAE={m['mae']:.2f} RMSE={m['rmse']:.2f} "
          f"MAPE={m['mape']:.1f}% bias={m['bias']:+.2f}")
    if "backtest_mape" in m:
        print(f"walk-forward backtest avg MAPE={m['backtest_mape']:.1f}% over {len(bt)} folds")
