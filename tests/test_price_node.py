"""测试议价节点。"""

import pytest
import unittest.mock
from unittest.mock import AsyncMock, patch, MagicMock
from agents.nodes.price import price_node, calculate_price_temperature
from agents.state import AgentState


@pytest.mark.asyncio
async def test_price_node_basic():
    """测试基本议价功能。"""
    mock_response = MagicMock()
    mock_response.content = "好的，这个价格我可以接受。"

    mock_llm_client = AsyncMock()
    mock_llm_client.invoke.return_value = mock_response

    with patch("agents.nodes.price.LLMClient", return_value=mock_llm_client), \
         patch("agents.nodes.price.check_safety", return_value="安全的回复"), \
         patch("builtins.open", unittest.mock.mock_open(read_data="price prompt")):

        state = AgentState(
            messages=["用户消息"],
            bargain_count=0,
            item_info={"min_price": "100"}
        )
        result = await price_node(state)

        assert len(result["messages"]) == 1
        assert result["bargain_count"] == 1


@pytest.mark.asyncio
async def test_price_node_file_not_found():
    """测试文件未找到的默认回复。"""
    with patch("builtins.open", side_effect=FileNotFoundError):
        state = AgentState(messages=["用户消息"], bargain_count=0, item_info={})
        result = await price_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "抱歉，暂时无法处理议价请求"


@pytest.mark.asyncio
async def test_price_node_llm_error():
    """测试LLM调用错误。"""
    mock_llm_client = AsyncMock()
    mock_llm_client.invoke.side_effect = Exception("LLM调用失败")

    with patch("agents.nodes.price.LLMClient", return_value=mock_llm_client), \
         patch("agents.nodes.price.check_safety", return_value="安全的回复"), \
         patch("builtins.open", unittest.mock.mock_open(read_data="price prompt")):

        state = AgentState(messages=["用户消息"], bargain_count=0, item_info={})
        result = await price_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "抱歉，处理议价请求时出错了"


def test_calculate_price_temperature():
    """测试温度计算函数。"""
    assert calculate_price_temperature(0) == pytest.approx(0.3)
    assert calculate_price_temperature(1) == pytest.approx(0.45)  # 使用近似比较避免浮点精度问题
    assert calculate_price_temperature(4) == pytest.approx(0.9)
    assert calculate_price_temperature(10) == pytest.approx(0.9)  # 超过最大值