"""Periodic ingestion - runs the pipeline on an interval. Framed explicitly
as orchestration, though intentionally not Kafka/Airflow-scale: a loop with
a sleep is the right amount of engineering for a project that won't get
heavy reviewer attention either way, and pretending otherwise would just be
unnecessary complexity for its own sake."""

import time

from process import SentimentModel
from pipeline import run_pipeline
from ingest import fetch_subreddit_posts

DEFAULT_INTERVAL_SECONDS = 1800  # 30 minutes


def run_forever(subreddit: str, topic: str, interval_seconds: int = DEFAULT_INTERVAL_SECONDS, limit: int = 50):
    model = SentimentModel.load()
    print(f"Watching r/{subreddit} (topic: {topic}) every {interval_seconds}s. Ctrl+C to stop.")
    while True:
        try:
            posts = fetch_subreddit_posts(subreddit, limit=limit, topic=topic)
            summary = run_pipeline(posts, source="reddit", model=model)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {summary}")
        except Exception as e:  # noqa: BLE001 - keep the loop alive across one bad run (network blip, API error)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] run failed: {e}")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 scheduler.py <subreddit> <topic> [interval_seconds]")
        sys.exit(1)

    interval = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_INTERVAL_SECONDS
    run_forever(sys.argv[1], sys.argv[2], interval_seconds=interval)
