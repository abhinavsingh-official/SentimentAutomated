"""Live ingestion from Reddit - replaces the static twitter_training.csv as
the primary data source. That CSV isn't deleted; load_batch_csv() below
keeps it available as an optional "run this on your own CSV" input mode,
for anyone who'd rather not set up Reddit API credentials just to try the
pipeline.

Needs a free Reddit "script" app (reddit.com/prefs/apps -> create app ->
type "script") - instant to register, no approval wait. Credentials go in
a local .env file (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET,
REDDIT_USER_AGENT), never committed."""

import os

import pandas as pd
import praw
from dotenv import load_dotenv

load_dotenv()


def reddit_client() -> praw.Reddit | None:
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "SentimentAutomated/0.1")
    if not client_id or not client_secret:
        return None
    return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)


def fetch_subreddit_posts(subreddit_name: str, limit: int = 100, topic: str | None = None) -> list[dict]:
    """Pulls the subreddit's newest posts (title + selftext combined as one
    text field) plus top-level comments - comments are often where the
    actual opinion/sentiment is, not just the post title."""
    client = reddit_client()
    if client is None:
        raise RuntimeError(
            "No Reddit API credentials found. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET "
            "in a .env file (see README for how to get a free one), or use load_batch_csv() "
            "for an offline demo instead."
        )

    topic = topic or subreddit_name
    subreddit = client.subreddit(subreddit_name)
    posts = []

    for submission in subreddit.new(limit=limit):
        text = f"{submission.title} {submission.selftext}".strip()
        posts.append({
            "post_id": submission.id,
            "text": text,
            "topic": topic,
            "created_at": pd.Timestamp(submission.created_utc, unit="s", tz="UTC").isoformat(),
            "url": submission.url,
        })

        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list()[:10]:
            if not comment.body or comment.body in ("[deleted]", "[removed]"):
                continue
            posts.append({
                "post_id": comment.id,
                "text": comment.body,
                "topic": topic,
                "created_at": pd.Timestamp(comment.created_utc, unit="s", tz="UTC").isoformat(),
                "url": f"https://reddit.com{comment.permalink}",
            })

    return posts


def load_batch_csv(path: str, text_column: str, topic: str, id_column: str | None = None) -> list[dict]:
    """Treats a static CSV as if it had just been "ingested" - the original
    twitter_training.csv can be run through the exact same
    quality -> process -> store pipeline as live Reddit data this way,
    useful both as a demo path and for testing the pipeline without
    needing live API credentials."""
    df = pd.read_csv(path)
    posts = []
    for i, row in df.iterrows():
        posts.append({
            "post_id": str(row[id_column]) if id_column else str(i),
            "text": str(row[text_column]),
            "topic": topic,
            "created_at": None,
        })
    return posts


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 ingest.py <subreddit> [limit]")
        sys.exit(1)

    subreddit_name = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    posts = fetch_subreddit_posts(subreddit_name, limit=limit)
    print(f"Fetched {len(posts)} posts/comments from r/{subreddit_name}")
