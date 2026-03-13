import time
import pytest
from unittest.mock import AsyncMock, patch
from auth.token_manager import TokenManager


@pytest.fixture
def tm():
    return TokenManager(cookies_str="test=1", device_id="dev-123")


def test_init(tm):
    assert tm.current_token is None
    assert tm.device_id == "dev-123"


def test_needs_refresh_initially(tm):
    assert tm.needs_refresh() is True


def test_needs_refresh_after_set(tm):
    tm.current_token = "tok123"
    tm.last_refresh_time = time.time()
    assert tm.needs_refresh() is False


def test_needs_refresh_after_timeout(tm):
    tm.current_token = "tok123"
    tm.last_refresh_time = time.time() - 4000  # > 3600
    assert tm.needs_refresh() is True
