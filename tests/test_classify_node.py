"""测试意图识别节点。"""

import pytest
import unittest.mock
from unittest.mock import AsyncMock, patch, MagicMock
from agents.nodes.classify import classify_node
from agents.state import AgentState


@pytest.mark.asyncio
async def test_classify_node_valid_intent():
    """测试识别有效意图。"""
    mock_response = MagicMock()
    mock_response.content = "price"

    mock_llm_client = AsyncMock()
    mock_llm_client.invoke.return_value = mock_response

    with (
        patch("agents.nodes.classify.LLMClient", return_value=mock_llm_client),
        patch("builtins.open", unittest.mock.mock_open(read_data="classify prompt")),
    ):
        state = AgentState(messages=["用户消息"])
        result = await classify_node(state)

        assert result["intent"] == "price"


@pytest.mark.asyncio
async def test_classify_node_invalid_intent():
    """测试无效意图默认处理。"""
    mock_response = MagicMock()
    mock_response.content = "invalid_intent"

    mock_llm_client = AsyncMock()
    mock_llm_client.invoke.return_value = mock_response

    with (
        patch("agents.nodes.classify.LLMClient", return_value=mock_llm_client),
        patch("builtins.open", unittest.mock.mock_open(read_data="classify prompt")),
    ):
        state = AgentState(messages=["用户消息"])
        result = await classify_node(state)

        assert result["intent"] == "default"


@pytest.mark.asyncio
async def test_classify_node_file_not_found():
    """测试文件未找到。"""
    with patch("builtins.open", side_effect=FileNotFoundError):
        state = AgentState(messages=["用户消息"])
        result = await classify_node(state)

        assert result["intent"] == "default"


@pytest.mark.asyncio
async def test_classify_node_llm_error():
    """测试LLM调用错误。"""
    mock_llm_client = AsyncMock()
    mock_llm_client.invoke.side_effect = Exception("LLM调用失败")

    with (
        patch("agents.nodes.classify.LLMClient", return_value=mock_llm_client),
        patch("builtins.open", unittest.mock.mock_open(read_data="classify prompt")),
    ):
        state = AgentState(messages=["用户消息"])
        result = await classify_node(state)

        assert result["intent"] == "default"
