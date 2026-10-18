"""Second ingestion source - NewsAPI's free tier, for the same tracked
topic as Reddit, so sentiment can be compared across source types (social
media opinion vs. news-headline framing) rather than treated as one
undifferentiated feed.

Needs a free NewsAPI key (newsapi.org/register, instant) in NEWSAPI_KEY."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_URL = "https://newsapi.org/v2/everything"


def fetch_news_headlines(query: str, topic: str | None = None, page_size: int = 50) -> list[dict]:
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError(
            "No NEWSAPI_KEY found. Get a free one at newsapi.org/register and set it in .env, "
            "or skip this source and use Reddit-only ingestion."
        )

    topic = topic or query
    response = requests.get(
        NEWSAPI_URL,
        params={"q": query, "pageSize": page_size, "language": "en", "sortBy": "publishedAt", "apiKey": api_key},
        timeout=10,
    )
    response.raise_for_status()
    articles = response.json().get("articles", [])

    posts = []
    for article in articles:
        text = " - ".join(filter(None, [article.get("title"), article.get("description")]))
        if not text:
            continue
        posts.append({
            "post_id": article.get("url", ""),  # NewsAPI has no numeric id; the URL is unique per article
            "text": text,
            "topic": topic,
            "created_at": article.get("publishedAt"),
            "url": article.get("url"),
        })
    return posts


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 news.py <search query>")
        sys.exit(1)

    posts = fetch_news_headlines(sys.argv[1])
    print(f"Fetched {len(posts)} articles for '{sys.argv[1]}'")
