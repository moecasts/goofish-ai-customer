"""测试商品咨询节点。"""

import pytest
import unittest.mock
from unittest.mock import AsyncMock, patch, MagicMock
from agents.nodes.product import product_node
from agents.state import AgentState


@pytest.mark.asyncio
async def test_product_node_basic():
    """测试基本商品咨询功能。"""
    mock_response = MagicMock()
    mock_response.content = "这款商品的特点是..."

    mock_llm_client = AsyncMock()
    mock_llm_client.invoke.return_value = mock_response

    with patch("agents.nodes.product.LLMClient", return_value=mock_llm_client), \
         patch("agents.nodes.product.check_safety", return_value="安全的回复"), \
         patch("builtins.open", unittest.mock.mock_open(read_data="product prompt")):

        state = AgentState(messages=["用户消息"])
        result = await product_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "安全的回复"


@pytest.mark.asyncio
async def test_product_node_file_not_found():
    """测试文件未找到的默认回复。"""
    with patch("builtins.open", side_effect=FileNotFoundError):
        state = AgentState(messages=["用户消息"])
        result = await product_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "抱歉，暂时无法处理商品咨询"


@pytest.mark.asyncio
async def test_product_node_llm_error():
    """测试LLM调用错误。"""
    mock_llm_client = AsyncMock()
    mock_llm_client.invoke.side_effect = Exception("LLM调用失败")

    with patch("agents.nodes.product.LLMClient", return_value=mock_llm_client), \
         patch("agents.nodes.product.check_safety", return_value="安全的回复"), \
         patch("builtins.open", unittest.mock.mock_open(read_data="product prompt")):

        state = AgentState(messages=["用户消息"])
        result = await product_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "抱歉，处理商品咨询时出错了"