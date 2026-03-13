import os
import tempfile
import pytest
from storage.context_manager import ContextManager


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    cm = ContextManager(path)
    yield cm
    cm.close()
    os.unlink(path)


def test_add_and_get_messages(db):
    db.add_message("chat1", "user1", "item1", "user", "hello")
    db.add_message("chat1", "user1", "item1", "assistant", "hi there")
    msgs = db.get_context("chat1")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["content"] == "hi there"


def test_get_context_empty(db):
    msgs = db.get_context("nonexistent")
    assert msgs == []


def test_bargain_count(db):
    assert db.get_bargain_count("chat1") == 0
    db.increment_bargain_count("chat1")
    assert db.get_bargain_count("chat1") == 1
    db.increment_bargain_count("chat1")
    assert db.get_bargain_count("chat1") == 2


def test_item_cache(db):
    db.save_item("item1", {"title": "iPhone", "price": 4500}, 4500, "test desc")
    item = db.get_item("item1")
    assert item["title"] == "iPhone"
    assert db.get_item("nonexistent") is None


def test_message_auto_cleanup(db):
    db.max_history = 5
    for i in range(10):
        db.add_message("chat1", "u1", "i1", "user", f"msg{i}")
    msgs = db.get_context("chat1")
    assert len(msgs) == 5
    assert msgs[0]["content"] == "msg5"
