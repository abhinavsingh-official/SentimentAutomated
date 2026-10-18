from topics import tag_by_keyword


def test_tag_by_keyword_matches_whole_word_only():
    keywords = {"ps": ["ps5"]}
    assert tag_by_keyword("I love my new ps5", keywords) == "ps"
    assert tag_by_keyword("oops that was a mistake", keywords) == "general"


def test_tag_by_keyword_returns_default_when_nothing_matches():
    keywords = {"xbox": ["xbox"]}
    assert tag_by_keyword("completely unrelated text", keywords, default="misc") == "misc"


def test_tag_by_keyword_is_case_insensitive():
    keywords = {"amazon": ["amazon"]}
    assert tag_by_keyword("I just ordered something from AMAZON", keywords) == "amazon"
