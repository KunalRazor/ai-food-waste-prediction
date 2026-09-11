FROM python:3.12-slim

WORKDIR /app

# libgomp1 is the OpenMP runtime XGBoost needs at import time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV MODEL_DIR=/app/models
EXPOSE 8000

# On first boot: generate data + train, then serve. See entrypoint.sh.
ENTRYPOINT ["./entrypoint.sh"]
