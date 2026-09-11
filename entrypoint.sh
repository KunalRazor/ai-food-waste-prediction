#!/bin/sh
set -e

# Seed data + train a model on first boot so the API has something to serve.
if [ ! -f "${MODEL_DIR:-./models}/current.json" ]; then
  echo "no model found -> generating synthetic data and training..."
  python -m app.data_gen
  python -m app.train
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
