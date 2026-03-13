import pytest
from unittest.mock import patch


@pytest.mark.unit
class TestTokenManager:
    """Test suite for TokenManager class."""

    def test_init(self, token_manager):
        """Test TokenManager initialization."""
        assert token_manager.current_token is None
        assert token_manager.device_id == "test-device-123"

    def test_needs_refresh_initially(self, token_manager):
        """Test that newly created manager needs refresh."""
        assert token_manager.needs_refresh() is True

    @pytest.mark.parametrize(
        "current_token,elapsed_seconds,expected",
        [
            ("tok123", 0, False),
            ("tok123", 1800, False),  # 30 minutes
            ("tok123", 4000, True),  # > 1 hour
            ("tok456", 7200, True),  # 2 hours
        ],
    )
    def test_needs_refresh_scenarios(
        self, token_manager, current_token, elapsed_seconds, expected
    ):
        """Test token refresh logic with various time scenarios."""
        with patch("time.time", return_value=elapsed_seconds):
            token_manager.current_token = current_token
            token_manager.last_refresh_time = 0
            assert token_manager.needs_refresh() is expected
