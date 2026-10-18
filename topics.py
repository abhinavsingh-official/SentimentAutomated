"""Lightweight topic/entity tagging - not full aspect-based sentiment
analysis, just enough to show sentiment broken down per topic instead of
one overall number. Two tagging strategies:

- A fixed keyword list per tracked topic (used for live ingestion, where
  you already know what you're monitoring - e.g. a subreddit about one
  product should tag everything with that product's name).
- spaCy noun-chunk extraction (used for exploring a batch of text with no
  predefined topic list - what are people actually talking about?)."""

import re
from collections import Counter

import spacy

_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP


def tag_by_keyword(text: str, topic_keywords: dict[str, list[str]], default: str = "general") -> str:
    """Returns the first topic whose keyword list matches text, case-insensitive
    whole-word matching so "ps" doesn't match inside "oops"."""
    lowered = text.lower()
    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword.lower())}\b", lowered):
                return topic
    return default


def top_noun_chunks(texts: list[str], top_n: int = 15, min_length: int = 3) -> list[tuple[str, int]]:
    """Runs noun-chunk extraction over a batch of text and returns the most
    common chunks - a cheap "what's actually being discussed" signal without
    a predefined topic list."""
    counts: Counter[str] = Counter()
    for doc in _nlp().pipe(texts, batch_size=64):
        for chunk in doc.noun_chunks:
            cleaned = chunk.text.strip().lower()
            if len(cleaned) >= min_length and not cleaned.isdigit():
                counts[cleaned] += 1
    return counts.most_common(top_n)
