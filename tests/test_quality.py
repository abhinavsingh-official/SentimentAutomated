from quality import run_quality_gate


def test_rejects_duplicate_post_id():
    posts = [{"post_id": "1", "text": "This is a perfectly good piece of English text"},
             {"post_id": "1", "text": "This is a different but also fine piece of text"}]
    result = run_quality_gate(posts)
    assert len(result.accepted) == 1
    assert result.rejected[0][1] == "duplicate post_id"


def test_rejects_near_empty_text():
    posts = [{"post_id": "1", "text": "ok"}]
    result = run_quality_gate(posts)
    assert result.accepted == []
    assert result.rejected[0][1] == "empty or near-empty text"


def test_rejects_non_english_text():
    posts = [{"post_id": "1", "text": "Ceci est un texte assez long écrit entièrement en français pour ce test"}]
    result = run_quality_gate(posts)
    assert result.accepted == []
    assert result.rejected[0][1] == "not English"


def test_accepts_and_truncates_long_english_text():
    # A long run of one repeated word (rather than varied real sentences)
    # confuses langdetect into rejecting it as non-English - this uses a
    # realistic repeated sentence instead, which is what truncation is
    # actually meant to handle.
    sentence = "This is a genuinely long English sentence about gaming and online multiplayer. "
    posts = [{"post_id": "1", "text": sentence * 15}]
    result = run_quality_gate(posts)
    assert len(result.accepted) == 1
    assert len(result.accepted[0]["text"].split()) <= 100


def test_accepts_normal_english_text_unchanged():
    text = "I really enjoyed playing this game over the weekend with friends"
    posts = [{"post_id": "1", "text": text}]
    result = run_quality_gate(posts)
    assert len(result.accepted) == 1
    assert result.accepted[0]["text"] == text
