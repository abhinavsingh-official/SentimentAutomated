import store


def test_save_posts_skips_duplicates(tmp_path):
    db_path = tmp_path / "test.db"
    post = store.ScoredPost(source="test", post_id="1", topic="demo", text="hello", sentiment="Positive", confidence=0.9)

    first = store.save_posts([post], db_path=db_path)
    second = store.save_posts([post], db_path=db_path)

    assert first == 1
    assert second == 0  # same (source, post_id) - not a new row
    assert store.stats(db_path)["total_posts"] == 1


def test_different_sources_same_post_id_both_stored(tmp_path):
    db_path = tmp_path / "test.db"
    reddit_post = store.ScoredPost(source="reddit", post_id="1", topic="demo", text="a", sentiment="Positive", confidence=0.9)
    news_post = store.ScoredPost(source="news", post_id="1", topic="demo", text="b", sentiment="Negative", confidence=0.8)

    store.save_posts([reddit_post, news_post], db_path=db_path)

    assert store.stats(db_path)["total_posts"] == 2


def test_log_rejection_and_recent_rejections(tmp_path):
    db_path = tmp_path / "test.db"
    store.log_rejection("reddit", "1", "empty or near-empty text", db_path=db_path)
    rejections = store.recent_rejections(db_path=db_path)
    assert len(rejections) == 1
    assert rejections[0]["reason"] == "empty or near-empty text"


def test_posts_by_topic_filters_correctly(tmp_path):
    db_path = tmp_path / "test.db"
    posts = [
        store.ScoredPost(source="test", post_id="1", topic="topic_a", text="a", sentiment="Positive", confidence=0.9),
        store.ScoredPost(source="test", post_id="2", topic="topic_b", text="b", sentiment="Negative", confidence=0.8),
    ]
    store.save_posts(posts, db_path=db_path)

    assert len(store.posts_by_topic("topic_a", db_path=db_path)) == 1
    assert len(store.posts_by_topic("topic_b", db_path=db_path)) == 1
    assert set(store.topics(db_path=db_path)) == {"topic_a", "topic_b"}
