"""意图路由器测试。"""

from agents.state import AgentState
from agents.graph import route_intent, check_bargain_continue


class MockMessage:
    """Mock消息类。"""

    def __init__(self, content):
        self.content = content


def test_route_intent_price():
    """测试价格意图路由。"""
    state = AgentState(
        messages=[],
        user_id="test",
        intent="price",
        bargain_count=0,
        item_info=None,
        manual_mode=False,
    )

    result = route_intent(state)
    assert result == "price"


def test_route_intent_product():
    """测试商品意图路由。"""
    state = AgentState(
        messages=[],
        user_id="test",
        intent="product",
        bargain_count=0,
        item_info=None,
        manual_mode=False,
    )

    result = route_intent(state)
    assert result == "product"


def test_route_intent_default():
    """测试默认意图路由。"""
    state = AgentState(
        messages=[],
        user_id="test",
        intent="default",
        bargain_count=0,
        item_info=None,
        manual_mode=False,
    )

    result = route_intent(state)
    assert result == "default"


def test_route_intent_no_reply():
    """测试无回复意图路由。"""
    state = AgentState(
        messages=[],
        user_id="test",
        intent="no_reply",
        bargain_count=0,
        item_info=None,
        manual_mode=False,
    )

    result = route_intent(state)
    assert result == "no_reply"


def test_route_intent_missing():
    """测试缺少意图时默认路由。"""
    state = AgentState(
        messages=[],
        user_id="test",
        intent=None,
        bargain_count=0,
        item_info=None,
        manual_mode=False,
    )

    result = route_intent(state)
    assert result == "default"


def test_check_bargain_continue_count_limit():
    """测试议价次数限制。"""
    state = AgentState(
        messages=[MockMessage("可以便宜点吗？")],
        user_id="test",
        intent="price",
        bargain_count=5,
        item_info=None,
        manual_mode=False,
    )

    result = check_bargain_continue(state)
    assert result == "stop"


def test_check_bargain_continue_price_keywords():
    """测试检测到价格关键词继续议价。"""
    state = AgentState(
        messages=[MockMessage("价格能再便宜点吗？")],
        user_id="test",
        intent="price",
        bargain_count=2,
        item_info=None,
        manual_mode=False,
    )

    result = check_bargain_continue(state)
    assert result == "continue"


def test_check_bargain_continue_no_price_keywords():
    """测试未检测到价格关键词停止议价。"""
    state = AgentState(
        messages=[MockMessage("这个商品怎么样？")],
        user_id="test",
        intent="price",
        bargain_count=2,
        item_info=None,
        manual_mode=False,
    )

    result = check_bargain_continue(state)
    assert result == "stop"


def test_check_bargain_continue_empty_content():
    """测试消息内容为空。"""
    state = AgentState(
        messages=[MockMessage("")],
        user_id="test",
        intent="price",
        bargain_count=2,
        item_info=None,
        manual_mode=False,
    )

    result = check_bargain_continue(state)
    assert result == "stop"


def test_check_bargain_continue_various_price_keywords():
    """测试各种价格关键词。"""
    test_cases = [
        ("这个多少钱？", "continue"),
        ("能砍价吗？", "continue"),
        ("便宜点", "continue"),
        ("最低价多少？", "continue"),
        ("可以少点吗？", "continue"),
        ("这个商品不错", "stop"),
        ("质量怎么样", "stop"),
        ("什么时候发货", "stop"),
    ]

    for content, expected in test_cases:
        state = AgentState(
            messages=[MockMessage(content)],
            user_id="test",
            intent="price",
            bargain_count=2,
            item_info=None,
            manual_mode=False,
        )

        result = check_bargain_continue(state)
        assert result == expected, f"Content: {content}"


def test_check_bargain_continue_case_insensitive():
    """测试大小写不敏感检测。"""
    state = AgentState(
        messages=[MockMessage("便宜点好吗？")],
        user_id="test",
        intent="price",
        bargain_count=2,
        item_info=None,
        manual_mode=False,
    )

    result = check_bargain_continue(state)
    assert result == "continue"