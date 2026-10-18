"""Ties ingest -> quality gate -> sentiment scoring -> topic tagging ->
storage into one call. scheduler.py drives this on an interval for live
ingestion; it's also what a one-off run (or the batch-CSV demo path) calls
directly."""

from process import SentimentModel, score_dataframe
from quality import run_quality_gate
from topics import tag_by_keyword

import pandas as pd

import store


def run_pipeline(
    posts: list[dict],
    source: str,
    model: SentimentModel | None = None,
    topic_keywords: dict[str, list[str]] | None = None,
) -> dict:
    """posts: list of dicts with post_id/text/topic/created_at (the shape
    ingest.py and news.py both produce). Returns a summary of what happened
    at each stage, so a caller (the scheduler, or a one-off script) can log
    or display it without re-deriving the same counts."""
    quality_result = run_quality_gate(posts)

    for post, reason in quality_result.rejected:
        store.log_rejection(source, post.get("post_id"), reason)

    if not quality_result.accepted:
        return {"ingested": len(posts), "accepted": 0, "rejected": len(quality_result.rejected), "stored": 0}

    df = pd.DataFrame(quality_result.accepted)
    scored = score_dataframe(df, text_column="text", model=model)

    if topic_keywords:
        scored["topic"] = scored["text"].apply(lambda t: tag_by_keyword(t, topic_keywords, default=scored["topic"].iloc[0]))

    scored_posts = [
        store.ScoredPost(
            source=source, post_id=str(row["post_id"]), topic=row["topic"], text=row["text"],
            sentiment=row["sentiment"], confidence=row["confidence"],
            created_at=row["created_at"] if pd.notna(row.get("created_at")) else None,
        )
        for _, row in scored.iterrows()
    ]
    inserted = store.save_posts(scored_posts)

    return {
        "ingested": len(posts),
        "accepted": len(quality_result.accepted),
        "rejected": len(quality_result.rejected),
        "stored": inserted,
    }
