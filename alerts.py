"""Threshold-based alerting over stored sentiment - a real "brand
monitoring" feature, not just a number on a dashboard. If the negative
share within a recent window crosses a configurable threshold, that's
flagged as an active alert the dashboard can surface."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import store

DEFAULT_THRESHOLD = 0.4  # negative-sentiment share that triggers an alert
DEFAULT_WINDOW_HOURS = 24
MIN_SAMPLE_SIZE = 5  # don't alert on a handful of posts - too noisy to mean anything


@dataclass
class Alert:
    topic: str
    negative_share: float
    sample_size: int
    window_hours: int
    threshold: float


def check_alerts(
    threshold: float = DEFAULT_THRESHOLD,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    db_path=store.DB_PATH,
) -> list[Alert]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
    alerts = []

    for topic in store.topics(db_path):
        posts = [p for p in store.posts_by_topic(topic, db_path) if p["ingested_at"] >= cutoff]
        if len(posts) < MIN_SAMPLE_SIZE:
            continue
        negative_count = sum(1 for p in posts if p["sentiment"] == "Negative")
        negative_share = negative_count / len(posts)
        if negative_share >= threshold:
            alerts.append(Alert(
                topic=topic, negative_share=negative_share, sample_size=len(posts),
                window_hours=window_hours, threshold=threshold,
            ))

    return alerts
