from hn_ingest import _strip_html


def test_strip_html_removes_tags():
    assert _strip_html("<p>hello <i>world</i></p>") == "hello  world"


def test_strip_html_unescapes_entities():
    assert _strip_html("a &amp; b &#x2F; c") == "a & b / c"


def test_strip_html_handles_plain_text_unchanged():
    assert _strip_html("just plain text") == "just plain text"
