import itertools
from datetime import datetime, timedelta, timezone

import store
from alerts import check_alerts

_post_id_counter = itertools.count()


def _recent_post(topic, sentiment, db_path, hours_ago=0):
    post = store.ScoredPost(
        source="test", post_id=f"{topic}-{sentiment}-{hours_ago}-{next(_post_id_counter)}",
        topic=topic, text="x", sentiment=sentiment, confidence=0.9,
    )
    store.save_posts([post], db_path=db_path)
    # save_posts always stamps "now" - backdate ingested_at directly for window tests.
    ingested_at = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    with store.connect(db_path) as conn:
        conn.execute("UPDATE posts SET ingested_at = ? WHERE post_id = ?", (ingested_at, post.post_id))


def test_alert_fires_when_negative_share_crosses_threshold(tmp_path):
    db_path = tmp_path / "test.db"
    for _ in range(6):
        _recent_post("brand_x", "Negative", db_path)
    for _ in range(2):
        _recent_post("brand_x", "Positive", db_path)

    alerts = check_alerts(threshold=0.5, db_path=db_path)
    assert len(alerts) == 1
    assert alerts[0].topic == "brand_x"
    assert alerts[0].negative_share == 0.75


def test_no_alert_below_threshold(tmp_path):
    db_path = tmp_path / "test.db"
    for _ in range(5):
        _recent_post("brand_x", "Positive", db_path)
    for _ in range(1):
        _recent_post("brand_x", "Negative", db_path)

    assert check_alerts(threshold=0.5, db_path=db_path) == []


def test_no_alert_below_minimum_sample_size(tmp_path):
    db_path = tmp_path / "test.db"
    for _ in range(3):
        _recent_post("brand_x", "Negative", db_path)

    # 100% negative, but only 3 posts - below MIN_SAMPLE_SIZE, shouldn't alert on noise this small.
    assert check_alerts(threshold=0.5, db_path=db_path) == []


def test_posts_outside_window_are_ignored(tmp_path):
    db_path = tmp_path / "test.db"
    for _ in range(6):
        _recent_post("brand_x", "Negative", db_path, hours_ago=48)

    assert check_alerts(threshold=0.5, window_hours=24, db_path=db_path) == []
