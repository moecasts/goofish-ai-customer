import pytest


@pytest.mark.integration
class TestContextManager:
    """Test suite for ContextManager (SQLite-based)."""

    def test_add_and_get_messages(self, context_manager):
        """Test adding messages and retrieving context."""
        context_manager.add_message("chat1", "user1", "item1", "user", "hello")
        context_manager.add_message("chat1", "user1", "item1", "assistant", "hi there")
        msgs = context_manager.get_context("chat1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["content"] == "hi there"

    def test_get_context_empty(self, context_manager):
        """Test getting context for non-existent chat returns empty list."""
        msgs = context_manager.get_context("nonexistent")
        assert msgs == []

    @pytest.mark.parametrize(
        "initial_count,increments,final_count",
        [
            (0, [1, 1], 2),
            (0, [1, 1, 1, 1, 1], 5),
            (3, [1], 4),
        ],
    )
    def test_bargain_count_tracking(
        self, context_manager, initial_count, increments, final_count
    ):
        """Test bargain count tracking with various increment patterns."""
        chat_id = "chat1"
        for _ in range(initial_count):
            context_manager.increment_bargain_count(chat_id)
        assert context_manager.get_bargain_count(chat_id) == initial_count

        for _ in increments:
            context_manager.increment_bargain_count(chat_id)
        assert context_manager.get_bargain_count(chat_id) == final_count

    def test_item_cache(self, context_manager):
        """Test item caching and retrieval."""
        context_manager.save_item(
            "item1", {"title": "iPhone", "price": 4500}, 4500, "test desc"
        )
        item = context_manager.get_item("item1")
        assert item["title"] == "iPhone"
        assert context_manager.get_item("nonexistent") is None

    @pytest.mark.parametrize(
        "max_history,add_count,expected_count,expected_first_msg",
        [
            (5, 10, 5, "msg5"),
            (3, 7, 3, "msg4"),
            (10, 5, 5, "msg0"),
        ],
    )
    def test_message_auto_cleanup(
        self,
        context_manager,
        max_history,
        add_count,
        expected_count,
        expected_first_msg,
    ):
        """Test automatic message cleanup when exceeding max_history."""
        context_manager.max_history = max_history
        for i in range(add_count):
            context_manager.add_message("chat1", "u1", "i1", "user", f"msg{i}")
        msgs = context_manager.get_context("chat1")
        assert len(msgs) == expected_count
        assert msgs[0]["content"] == expected_first_msg
