"""测试默认回复节点。"""

import pytest
import unittest.mock
from unittest.mock import AsyncMock, patch, MagicMock
from agents.nodes.default import default_node
from agents.state import AgentState


@pytest.mark.asyncio
async def test_default_node_basic():
    """测试基本默认回复功能。"""
    mock_response = MagicMock()
    mock_response.content = "您好，请问有什么可以帮您的？"

    mock_llm_client = AsyncMock()
    mock_llm_client.invoke.return_value = mock_response

    with patch("agents.nodes.default.LLMClient", return_value=mock_llm_client), \
         patch("agents.nodes.default.check_safety", return_value="安全的回复"), \
         patch("builtins.open", unittest.mock.mock_open(read_data="default prompt")):

        state = AgentState(messages=["用户消息"])
        result = await default_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "安全的回复"


@pytest.mark.asyncio
async def test_default_node_file_not_found():
    """测试文件未找到的默认回复。"""
    with patch("builtins.open", side_effect=FileNotFoundError):
        state = AgentState(messages=["用户消息"])
        result = await default_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "您好，请问有什么可以帮您的？"


@pytest.mark.asyncio
async def test_default_node_llm_error():
    """测试LLM调用错误。"""
    mock_llm_client = AsyncMock()
    mock_llm_client.invoke.side_effect = Exception("LLM调用失败")

    with patch("agents.nodes.default.LLMClient", return_value=mock_llm_client), \
         patch("agents.nodes.default.check_safety", return_value="安全的回复"), \
         patch("builtins.open", unittest.mock.mock_open(read_data="default prompt")):

        state = AgentState(messages=["用户消息"])
        result = await default_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "抱歉，我现在无法回复，请稍后再试"