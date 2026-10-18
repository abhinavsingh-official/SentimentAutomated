"""Data quality gate - runs before anything gets scored or stored. Mirrors
the validate/reject-and-log pattern used in the data-infra-learn flagship
project: every rejection gets a reason logged, not silently dropped, so you
can tell later whether the pipeline is actually filtering junk or losing
good data to an overly strict rule."""

from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect

# langdetect is non-deterministic by default (it samples n-grams); fixing
# the seed makes repeated runs on the same text give the same language call.
DetectorFactory.seed = 0

MIN_TEXT_LENGTH = 10
MAX_TOKENS = 100  # truncate longer text rather than let the tokenizer error


@dataclass
class QualityResult:
    accepted: list[dict]
    rejected: list[tuple[dict, str]]  # (post, reason)


def _is_near_empty(text: str) -> bool:
    return len(text.strip()) < MIN_TEXT_LENGTH


def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        # langdetect can't call a language on very short/ambiguous text -
        # treat "couldn't tell" as a rejection reason of its own, not a pass.
        return False


def _truncate(text: str, max_tokens: int = MAX_TOKENS) -> str:
    words = text.split()
    return " ".join(words[:max_tokens]) if len(words) > max_tokens else text


def run_quality_gate(posts: list[dict], text_field: str = "text", id_field: str = "post_id") -> QualityResult:
    """posts: list of dicts with at least text_field and id_field.
    Checks in order: dedupe by id -> reject empty/near-empty -> reject
    non-English -> truncate long text. Each rejection stops at the first
    reason that applies, so one row only ever contributes one rejection
    log entry, not several."""
    accepted = []
    rejected = []
    seen_ids = set()

    for post in posts:
        post_id = post.get(id_field)
        text = str(post.get(text_field, ""))

        if post_id in seen_ids:
            rejected.append((post, "duplicate post_id"))
            continue
        seen_ids.add(post_id)

        if _is_near_empty(text):
            rejected.append((post, "empty or near-empty text"))
            continue

        if not _is_english(text):
            rejected.append((post, "not English"))
            continue

        post = dict(post)
        post[text_field] = _truncate(text)
        accepted.append(post)

    return QualityResult(accepted=accepted, rejected=rejected)
