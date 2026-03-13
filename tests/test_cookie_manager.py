import pytest
from auth.cookie_manager import CookieManager


@pytest.mark.unit
class TestCookieManager:
    """Test suite for CookieManager class."""

    def test_load_from_env(self, cookie_manager, monkeypatch):
        """Test loading cookies from environment variable."""
        monkeypatch.setenv("COOKIES_STR", "a=1; b=2")
        cm = CookieManager(data_dir=str(cookie_manager.data_dir))
        assert cm.get_cookies_str() == "a=1; b=2"

    def test_save_and_load_json(self, cookie_manager):
        """Test saving and loading cookies from JSON file."""
        cookie_manager.update_cookies("x=10; y=20")
        # Reload to verify persistence
        cm2 = CookieManager(data_dir=str(cookie_manager.data_dir))
        assert "x=10" in cm2.get_cookies_str()

    @pytest.mark.parametrize(
        "cookie_str,expected",
        [
            ("a=1; b=2; c=3", {"a": "1", "b": "2", "c": "3"}),
            ("single=val", {"single": "val"}),
            ("key1=val1; key2=val2", {"key1": "val1", "key2": "val2"}),
        ],
    )
    def test_parse_cookies(self, cookie_manager, cookie_str, expected):
        """Test cookie string parsing with various formats."""
        result = cookie_manager._parse_cookie_str(cookie_str)
        assert result == expected
