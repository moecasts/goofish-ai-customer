"""Edge case and error handling tests for various components."""

import pytest
from auth.cookie_manager import CookieManager
from services.xianyu_utils import decrypt_message


@pytest.mark.unit
class TestEdgeCases:
    """Edge case tests for various components."""

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "",  # Empty string
            "   ",  # Whitespace only
            "a=1; =2",  # Missing key
            "=value",  # Missing value
            "key=",  # Empty value
            ";",  # Just separator
            "a=1;;b=2",  # Double separator
        ],
    )
    def test_parse_cookies_invalid_inputs(self, invalid_input):
        """Test cookie parsing with invalid inputs."""
        cm = CookieManager.__new__(CookieManager)
        result = cm._parse_cookie_str(invalid_input)
        assert isinstance(result, dict)

    def test_parse_cookies_with_special_characters(self):
        """Test cookie parsing with special characters in values."""
        cm = CookieManager.__new__(CookieManager)
        result = cm._parse_cookie_str(
            "key1=value%20with%20spaces; key2=value=with=equals"
        )
        assert result["key1"] == "value%20with%20spaces"
        assert result["key2"] == "value=with=equals"

    @pytest.mark.parametrize(
        "user_id",
        [
            "123",  # Short user ID
            "a" * 100,  # Very long user ID
            "user-with-dashes",  # User ID with dashes
            "user_with_underscores",  # User ID with underscores
        ],
    )
    def test_generate_device_id_edge_cases(self, user_id):
        """Test device ID generation with edge case user IDs."""
        from services.xianyu_utils import generate_device_id
        import re

        did = generate_device_id(user_id)
        assert did.endswith(f"-{user_id}")
        uuid_part = did.replace(f"-{user_id}", "")
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", uuid_part
        )

    def test_decrypt_invalid_base64(self):
        """Test decrypting invalid base64 data handles gracefully."""
        # The decrypt_message function handles various inputs
        result = decrypt_message("not-valid-base64!!!")
        # Should handle gracefully without crashing - can return list, dict, int, or other types
        assert result is not None

    def test_decrypt_non_json_base64(self):
        """Test decrypting base64 data that's not JSON."""
        import base64

        encoded = base64.b64encode(b"not json data").decode()
        result = decrypt_message(encoded)
        # Should handle gracefully - may return various types depending on parsing
        assert result is not None

    def test_decrypt_empty_string(self):
        """Test decrypting empty string."""
        result = decrypt_message("")
        # Should handle gracefully
        assert result is None or isinstance(result, dict)

    @pytest.mark.parametrize(
        "timestamp,token,data",
        [
            ("", "tok", "data"),  # Empty timestamp
            ("123", "", "data"),  # Empty token
            ("123", "tok", ""),  # Empty data
            ("invalid", "tok", "data"),  # Invalid timestamp
        ],
    )
    def test_sign_edge_cases(self, timestamp, token, data):
        """Test signature generation with edge case inputs."""
        from services.xianyu_utils import generate_sign

        sign = generate_sign(timestamp, token, data)
        # Should always return 32-character hex string
        assert len(sign) == 32
        assert all(c in "0123456789abcdef" for c in sign)

    def test_context_manager_nonexistent_chat(self):
        """Test getting bargain count for non-existent chat."""
        from storage.context_manager import ContextManager
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        cm = ContextManager(db_path)
        try:
            assert cm.get_bargain_count("nonexistent_chat") == 0
        finally:
            cm.close()
            import os

            os.unlink(db_path)

    def test_context_manager_empty_item_cache(self):
        """Test getting non-existent item from cache."""
        from storage.context_manager import ContextManager
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        cm = ContextManager(db_path)
        try:
            assert cm.get_item("nonexistent_item") is None
        finally:
            cm.close()
            os.unlink(db_path)


@pytest.mark.unit
class TestTokenManagerEdgeCases:
    """Edge case tests for TokenManager."""

    def test_token_manager_with_empty_device_id(self):
        """Test TokenManager with empty device ID."""
        from auth.token_manager import TokenManager

        tm = TokenManager(cookies_str="test=1", device_id="")
        assert tm.device_id == ""

    def test_token_manager_with_empty_cookies(self):
        """Test TokenManager with empty cookies string."""
        from auth.token_manager import TokenManager

        tm = TokenManager(cookies_str="", device_id="test-device")
        assert tm.device_id == "test-device"


@pytest.mark.unit
class TestWebSocketChannelEdgeCases:
    """Edge case tests for WebSocketChannel."""

    def test_parse_chat_message_missing_fields(self):
        """Test parsing chat message with missing required fields."""
        from core.websocket_channel import WebSocketChannel

        ch = WebSocketChannel.__new__(WebSocketChannel)

        # Missing senderUserId
        msg = {
            "1": {
                "2": "chat123@goofish_extra",
                "5": 1710000000000,
                "10": {
                    "reminderContent": "hello",
                },
            }
        }
        result = ch._parse_chat_message(msg)
        # When senderUserId is missing, it defaults to empty string
        assert result["sender_id"] == ""
        assert result["chat_id"] == "chat123"

    def test_parse_chat_message_empty_url(self):
        """Test parsing chat message with empty reminder URL."""
        from core.websocket_channel import WebSocketChannel

        ch = WebSocketChannel.__new__(WebSocketChannel)

        msg = {
            "1": {
                "2": "chat123@goofish_extra",
                "5": 1710000000000,
                "10": {
                    "senderUserId": "buyer456",
                    "reminderContent": "hello",
                    "reminderUrl": "",
                },
            }
        }
        result = ch._parse_chat_message(msg)
        assert result["chat_id"] == "chat123"
        # item_id should be None or empty when URL is empty
        assert not result.get("item_id")


@pytest.mark.unit
class TestAgentEdgeCases:
    """Edge case tests for agent components."""

    @pytest.mark.parametrize(
        "query",
        [
            "",  # Empty string
            "   ",  # Whitespace only
            "!!!",  # Special characters only
            "a" * 1000,  # Very long string
        ],
    )
    def test_intent_router_edge_cases(self, query):
        """Test IntentRouter with edge case queries."""
        from agents.router import IntentRouter

        router = IntentRouter()
        result = router.keyword_match(query)
        # Should handle edge cases gracefully
        assert result is None or result in ["price", "product"]

    def test_price_agent_negative_round(self):
        """Test PriceAgent with negative round number."""
        from agents.price_agent import PriceAgent

        agent = PriceAgent()
        temp = agent.get_temperature(-1)
        # Should handle negative gracefully
        assert isinstance(temp, float)

    def test_price_agent_very_large_round(self):
        """Test PriceAgent with very large round number."""
        from agents.price_agent import PriceAgent

        agent = PriceAgent()
        temp = agent.get_temperature(99999)
        # Should cap at maximum
        assert temp == pytest.approx(0.9)
