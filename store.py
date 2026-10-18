"""SQLite storage for scored posts - replaces always-overwriting
results.csv with an append-only table, so running ingestion repeatedly
builds up history instead of clobbering the previous run's output."""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("sentiment.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    post_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    text TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    confidence REAL NOT NULL,
    entities TEXT,
    created_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    UNIQUE(source, post_id)
);
CREATE INDEX IF NOT EXISTS idx_posts_topic ON posts(topic);
CREATE INDEX IF NOT EXISTS idx_posts_ingested_at ON posts(ingested_at);

CREATE TABLE IF NOT EXISTS rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    post_id TEXT,
    reason TEXT NOT NULL,
    rejected_at TEXT NOT NULL
);
"""


@dataclass
class ScoredPost:
    source: str
    post_id: str
    topic: str
    text: str
    sentiment: str
    confidence: float
    entities: str = ""
    created_at: str | None = None


@contextmanager
def connect(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_posts(posts: list[ScoredPost], db_path: Path = DB_PATH) -> int:
    """Inserts posts, skipping any (source, post_id) pair already stored -
    running ingestion again on overlapping data is a no-op for repeats,
    not a duplicate row."""
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    with connect(db_path) as conn:
        for post in posts:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO posts
                   (source, post_id, topic, text, sentiment, confidence, entities, created_at, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (post.source, post.post_id, post.topic, post.text, post.sentiment,
                 post.confidence, post.entities, post.created_at or now, now),
            )
            inserted += cursor.rowcount
    return inserted


def log_rejection(source: str, post_id: str | None, reason: str, db_path: Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO rejections (source, post_id, reason, rejected_at) VALUES (?, ?, ?, ?)",
            (source, post_id, reason, datetime.now(timezone.utc).isoformat()),
        )


def all_posts(db_path: Path = DB_PATH):
    with connect(db_path) as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM posts ORDER BY created_at DESC")]


def posts_by_topic(topic: str, db_path: Path = DB_PATH):
    with connect(db_path) as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM posts WHERE topic = ? ORDER BY created_at", (topic,)
        )]


def topics(db_path: Path = DB_PATH) -> list[str]:
    with connect(db_path) as conn:
        return [row["topic"] for row in conn.execute("SELECT DISTINCT topic FROM posts")]


def recent_rejections(limit: int = 50, db_path: Path = DB_PATH):
    with connect(db_path) as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM rejections ORDER BY rejected_at DESC LIMIT ?", (limit,)
        )]


def stats(db_path: Path = DB_PATH) -> dict:
    with connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        rejected = conn.execute("SELECT COUNT(*) FROM rejections").fetchone()[0]
        sources = [row["source"] for row in conn.execute("SELECT DISTINCT source FROM posts")]
        topic_count = conn.execute("SELECT COUNT(DISTINCT topic) FROM posts").fetchone()[0]
    return {"total_posts": total, "total_rejected": rejected, "sources": sources, "topic_count": topic_count}
