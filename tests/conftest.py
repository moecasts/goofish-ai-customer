"""Shared fixtures for pytest tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Generator, AsyncGenerator

import pytest


@pytest.fixture
def test_config_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test response"))]
    )
    return mock_client


@pytest.fixture
def mock_httpx_client() -> MagicMock:
    """Create a mock httpx async client."""
    mock_client = MagicMock()
    mock_client.get = AsyncMock()
    mock_client.post = AsyncMock()
    mock_client.get.return_value = MagicMock(status_code=200, json=lambda: {})
    mock_client.post.return_value = MagicMock(status_code=200, json=lambda: {})
    return mock_client


@pytest.fixture
async def mock_websocket() -> AsyncGenerator[MagicMock, None]:
    """Create a mock WebSocket connection."""
    mock_ws = MagicMock()
    mock_ws.send = AsyncMock()
    mock_ws.recv = AsyncMock(return_value='{"type":"test","data":"message"}')
    mock_ws.close = AsyncMock()
    yield mock_ws


@pytest.fixture
def sample_cookies() -> dict[str, str]:
    """Sample cookies for testing."""
    return {
        "cookie1": "value1",
        "cookie2": "value2",
        "session_id": "test_session_123",
    }


@pytest.fixture
def sample_chat_message() -> dict:
    """Sample chat message for testing."""
    return {
        "type": "chat",
        "content": "Hello, this is a test message",
        "sender": "user_123",
        "timestamp": "2024-01-01T00:00:00Z",
    }


@pytest.fixture(autouse=True)
def reset_environment() -> Generator[None, None, None]:
    """Automatically reset environment variables before each test."""
    # Save original environment
    original_env = os.environ.copy()

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_env_variables(test_config_dir: Path) -> None:
    """Set up common environment variables for testing."""
    os.environ["CONFIG_DIR"] = str(test_config_dir)
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["OPENAI_API_KEY"] = "test_api_key"


@pytest.fixture
def sample_config_yaml(test_config_dir: Path) -> Path:
    """Create a sample config.yaml file for testing."""
    config_path = test_config_dir / "config.yaml"
    config_content = """
api:
  base_url: "https://api.example.com"
  timeout: 30
  retry: 3

websocket:
  url: "ws://localhost:8080/ws"
  reconnect_interval: 5

logging:
  level: "INFO"
  format: "json"
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def device_id() -> str:
    """Standard device ID for testing."""
    return "test-device-123"


@pytest.fixture
def app_key_ws() -> str:
    """App key for WebSocket tests."""
    return "444e9908a51d1cb236a27862abc769c9"


@pytest.fixture
def app_key_api() -> str:
    """App key for API tests."""
    return "34839810"


@pytest.fixture
def seller_id() -> str:
    """Standard seller user ID for testing."""
    return "seller123"


@pytest.fixture
def buyer_id() -> str:
    """Standard buyer user ID for testing."""
    return "buyer789"


@pytest.fixture
def chat_id() -> str:
    """Standard chat ID for testing."""
    return "chat456"


@pytest.fixture
def sample_cookies_str() -> str:
    """Sample cookie string for testing."""
    return "cookie1=value1; cookie2=value2; session_id=test_session_123"


@pytest.fixture
def token_manager(device_id):
    """Factory fixture creating TokenManager instances."""
    from auth.token_manager import TokenManager

    return TokenManager(cookies_str="test=1", device_id=device_id)


@pytest.fixture
def cookie_manager(test_data_dir):
    """Factory fixture creating CookieManager instances."""
    from auth.cookie_manager import CookieManager

    return CookieManager(data_dir=str(test_data_dir))


@pytest.fixture
def websocket_channel(device_id):
    """Create WebSocketChannel instance for testing."""
    from core.websocket_channel import WebSocketChannel

    ch = WebSocketChannel.__new__(WebSocketChannel)
    ch.token = "test_token"
    ch.device_id = device_id
    ch.my_id = "seller123"
    return ch


@pytest.fixture
def context_manager(test_data_dir):
    """Create ContextManager instance for testing."""
    from storage.context_manager import ContextManager

    db_path = test_data_dir / "test.db"
    cm = ContextManager(str(db_path))
    yield cm
    cm.close()


@pytest.fixture
def intent_router():
    """Create IntentRouter instance for testing."""
    from agents.router import IntentRouter

    return IntentRouter()


@pytest.fixture
def price_agent():
    """Create PriceAgent instance for testing."""
    from agents.price_agent import PriceAgent

    return PriceAgent()
