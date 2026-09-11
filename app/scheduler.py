"""Scheduler: runs the daily batch and a weekly retrain.

In production this maps to a Kubernetes CronJob per task; APScheduler keeps the
same idea runnable in one process for a demo. Start with:

    python -m app.scheduler
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from .batch import run_daily_batch
from .train import train_and_register

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scheduler")


def daily_job() -> None:
    log.info("daily batch: %s", run_daily_batch())


def weekly_retrain() -> None:
    version, m, _ = train_and_register()
    log.info("retrained model=%s MAPE=%.1f%%", version, m["mape"])


def main() -> None:
    sched = BlockingScheduler(timezone="UTC")
    # Every day at 03:07 UTC (off the top of the hour on purpose).
    sched.add_job(daily_job, "cron", hour=3, minute=7, id="daily_batch")
    # Every Monday at 02:23 UTC.
    sched.add_job(weekly_retrain, "cron", day_of_week="mon", hour=2, minute=23, id="weekly_retrain")
    log.info("scheduler started (daily batch + weekly retrain)")
    sched.start()


if __name__ == "__main__":
    main()
