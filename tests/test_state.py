"""测试 Agent 状态定义。"""

import pytest
from agents.state import AgentState, ItemInfo
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import ValidationError


def test_item_info_creation():
    """测试商品信息创建。"""
    info = ItemInfo(
        item_id="123",
        title="测试商品",
        price=100.0,
        min_price=80.0,
    )
    assert info.item_id == "123"
    assert info.title == "测试商品"
    assert info.price == 100.0
    assert info.min_price == 80.0


def test_item_info_optional_fields():
    """测试商品信息可选字段。"""
    info = ItemInfo(
        item_id="456",
        title="另一个商品",
        price=50.0,
    )
    assert info.item_id == "456"
    assert info.title == "另一个商品"
    assert info.price == 50.0
    assert info.min_price is None
    assert info.description is None


def test_item_info_validation_negative_price():
    """测试商品信息价格验证 - 负价格。"""
    with pytest.raises(ValidationError):
        ItemInfo(
            item_id="789",
            title="无效商品",
            price=-10.0,
        )


def test_item_info_validation_zero_price():
    """测试商品信息价格验证 - 零价格。"""
    with pytest.raises(ValidationError):
        ItemInfo(
            item_id="790",
            title="无效商品",
            price=0.0,
        )


def test_item_info_validation_negative_min_price():
    """测试商品信息最低价验证 - 负最低价。"""
    with pytest.raises(ValidationError):
        ItemInfo(
            item_id="791",
            title="无效商品",
            price=100.0,
            min_price=-5.0,
        )


def test_item_info_validation_empty_title():
    """测试商品信息标题验证 - 空标题。"""
    with pytest.raises(ValidationError):
        ItemInfo(
            item_id="792",
            title="",
            price=100.0,
        )


def test_agent_state_structure():
    """测试 AgentState 结构。"""
    state = AgentState(
        messages=[HumanMessage(content="测试消息")],
        user_id="test_user",
        intent="price",
        bargain_count=1,
        item_info=None,
        manual_mode=False,
    )
    assert len(state["messages"]) == 1
    assert state["user_id"] == "test_user"
    assert state["intent"] == "price"
    assert state["bargain_count"] == 1


def test_agent_state_with_empty_messages():
    """测试 AgentState 空消息列表。"""
    state = AgentState(
        messages=[],
        user_id="test_user",
        intent="greeting",
        bargain_count=0,
        item_info=None,
        manual_mode=False,
    )
    assert len(state["messages"]) == 0
    assert state["bargain_count"] == 0


def test_agent_state_with_item_info():
    """测试 AgentState 包含商品信息。"""
    info = ItemInfo(
        item_id="999",
        title="完整商品",
        price=200.0,
        min_price=150.0,
        description="商品描述",
    )
    state = AgentState(
        messages=[
            HumanMessage(content="查询商品"),
            AIMessage(content="这是商品信息"),
        ],
        user_id="user123",
        intent="product",
        bargain_count=2,
        item_info=info,
        manual_mode=True,
    )
    assert len(state["messages"]) == 2
    assert state["item_info"].item_id == "999"
    assert state["item_info"].title == "完整商品"
    assert state["manual_mode"] is True
