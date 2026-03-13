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
