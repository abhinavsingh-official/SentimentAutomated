"""Live ingestion from Hacker News - the primary live source, swapped in
for Reddit after Reddit's API access moved to a manual-approval process
this year (no longer the instant self-serve registration the original
plan assumed - confirmed by trying it directly). HN's Algolia-backed
search API is fully public: no key, no registration, no approval, and it
returns the same shape of data (real user-written text with a timestamp)
this pipeline needs.

ingest.py's Reddit path is kept, not deleted - it still works for anyone
who does have Reddit API access, and the pipeline/quality-gate/storage
code downstream doesn't care which source a post dict came from."""

import requests

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"


def fetch_hn_comments(query: str, topic: str | None = None, hits: int = 50) -> list[dict]:
    """Searches HN comments (not just story titles - comments are where
    actual opinion/sentiment lives, same reasoning as ingest.py's Reddit
    path pulling comments alongside post titles) for `query`, most recent
    first."""
    topic = topic or query
    response = requests.get(
        SEARCH_URL,
        params={"query": query, "tags": "comment", "hitsPerPage": hits},
        timeout=10,
    )
    response.raise_for_status()
    hits_data = response.json().get("hits", [])

    posts = []
    for hit in hits_data:
        text = hit.get("comment_text")
        if not text:
            continue
        posts.append({
            "post_id": str(hit["objectID"]),
            "text": _strip_html(text),
            "topic": topic,
            "created_at": hit.get("created_at"),
            "url": f"https://news.ycombinator.com/item?id={hit['objectID']}",
        })
    return posts


def _strip_html(text: str) -> str:
    """HN comment text comes back with basic HTML (<p>, <i>, escaped
    entities) - strip it down to plain text rather than feeding tags to
    the sentiment model and quality gate."""
    import html
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text).strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 hn_ingest.py <search query>")
        sys.exit(1)

    posts = fetch_hn_comments(sys.argv[1])
    print(f"Fetched {len(posts)} comments for '{sys.argv[1]}'")
