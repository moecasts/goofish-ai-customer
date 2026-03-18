"""测试 skill_executor 节点。"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents.nodes.skill_executor import make_skill_executor
from agents.skill_registry import Skill, SkillRegistry
from agents.state import AgentState


def make_mock_registry(skill: Skill) -> MagicMock:
    """创建返回指定 skill 的 mock registry。"""
    registry = MagicMock(spec=SkillRegistry)
    registry.get_skill.return_value = skill
    return registry


@pytest.fixture
def price_skill() -> Skill:
    return Skill(
        name="price",
        description="处理议价",
        prompt="议价 prompt，底价 {min_price}，次数 {bargain_count}",
        state_hooks=["bargain_count", "min_price"],
        write_hooks=["bargain_count"],
        skill_dir=Path("/tmp/price"),
    )


@pytest.fixture
def product_skill() -> Skill:
    return Skill(
        name="product",
        description="处理商品咨询",
        prompt="商品咨询 prompt",
        state_hooks=[],
        write_hooks=[],
        skill_dir=Path("/tmp/product"),
    )


@pytest.mark.asyncio
async def test_executor_injects_state_hooks(price_skill):
    """测试 state_hooks 变量被正确注入 prompt。"""
    registry = make_mock_registry(price_skill)
    mock_response = MagicMock()
    mock_response.content = "好的，这个价格可以"
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    state = AgentState(
        messages=[HumanMessage(content="能便宜点吗")],
        user_id="u1",
        intent="price",
        bargain_count=1,
        item_info={"min_price": "100", "product_name": "手机"},
        manual_mode=False,
    )

    with patch("agents.nodes.skill_executor.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor.check_safety", side_effect=lambda x: x), \
         patch("agents.nodes.skill_executor._GLOBAL_RULES_PATH") as mock_rules_path:
        mock_rules_path.read_text.return_value = ""
        executor = make_skill_executor(registry)
    result = await executor(state)

    # 验证 LLM 被调用时 prompt 包含注入的变量值
    call_args = mock_llm.invoke.call_args[0][0]
    system_msg = call_args[0]
    assert isinstance(system_msg, SystemMessage)
    assert "100" in system_msg.content    # min_price 注入
    assert "1" in system_msg.content      # bargain_count 注入


@pytest.mark.asyncio
async def test_executor_increments_bargain_count(price_skill):
    """测试 price skill 的 bargain_count 在执行后 +1。"""
    registry = make_mock_registry(price_skill)
    mock_response = MagicMock()
    mock_response.content = "价格不能再低了"
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    state = AgentState(
        messages=[HumanMessage(content="再便宜一点")],
        user_id="u1",
        intent="price",
        bargain_count=2,
        item_info={"min_price": "80"},
        manual_mode=False,
    )

    with patch("agents.nodes.skill_executor.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor.check_safety", side_effect=lambda x: x), \
         patch("agents.nodes.skill_executor._GLOBAL_RULES_PATH") as mock_rules_path:
        mock_rules_path.read_text.return_value = ""
        executor = make_skill_executor(registry)
    result = await executor(state)

    assert result["bargain_count"] == 3


@pytest.mark.asyncio
async def test_executor_no_bargain_count_for_product(product_skill):
    """测试 product skill 执行后不修改 bargain_count。"""
    registry = make_mock_registry(product_skill)
    mock_response = MagicMock()
    mock_response.content = "这款商品..."
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    state = AgentState(
        messages=[HumanMessage(content="这个商品多大？")],
        user_id="u1",
        intent="product",
        bargain_count=0,
        item_info={},
        manual_mode=False,
    )

    with patch("agents.nodes.skill_executor.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor.check_safety", side_effect=lambda x: x), \
         patch("agents.nodes.skill_executor._GLOBAL_RULES_PATH") as mock_rules_path:
        mock_rules_path.read_text.return_value = ""
        executor = make_skill_executor(registry)
    result = await executor(state)

    assert "bargain_count" not in result


@pytest.mark.asyncio
async def test_executor_applies_safety_filter(product_skill):
    """测试 LLM 输出经过 check_safety 过滤。"""
    registry = make_mock_registry(product_skill)
    mock_response = MagicMock()
    mock_response.content = "加我微信"
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    state = AgentState(
        messages=[HumanMessage(content="怎么联系你？")],
        user_id="u1",
        intent="product",
        bargain_count=0,
        item_info={},
        manual_mode=False,
    )

    with patch("agents.nodes.skill_executor.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor._GLOBAL_RULES_PATH") as mock_rules_path:
        mock_rules_path.read_text.return_value = ""
        executor = make_skill_executor(registry)
    with patch("agents.nodes.skill_executor.check_safety", return_value="[安全提醒]"):
        result = await executor(state)

    assert result["messages"][-1].content == "[安全提醒]"


@pytest.mark.asyncio
async def test_executor_unknown_skill_falls_back_to_default(product_skill):
    """测试未知 skill name 时 registry 返回 None，executor 使用 fallback 回复。"""
    registry = MagicMock(spec=SkillRegistry)
    registry.get_skill.return_value = None  # 未知 skill

    state = AgentState(
        messages=[HumanMessage(content="随便说点什么")],
        user_id="u1",
        intent="unknown_skill",
        bargain_count=0,
        item_info={},
        manual_mode=False,
    )

    with patch("agents.nodes.skill_executor._GLOBAL_RULES_PATH") as mock_rules_path:
        mock_rules_path.read_text.return_value = ""
        executor = make_skill_executor(registry)
    result = await executor(state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)


@pytest.mark.asyncio
async def test_executor_llm_error_returns_fallback(price_skill):
    """测试 LLM 调用失败时返回 fallback 回复。"""
    registry = make_mock_registry(price_skill)
    mock_llm = AsyncMock()
    mock_llm.invoke.side_effect = Exception("LLM 挂了")

    state = AgentState(
        messages=[HumanMessage(content="能便宜吗")],
        user_id="u1",
        intent="price",
        bargain_count=0,
        item_info={"min_price": "50"},
        manual_mode=False,
    )

    with patch("agents.nodes.skill_executor.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor.check_safety", side_effect=lambda x: x), \
         patch("agents.nodes.skill_executor._GLOBAL_RULES_PATH") as mock_rules_path:
        mock_rules_path.read_text.return_value = ""
        executor = make_skill_executor(registry)
    result = await executor(state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)


@pytest.mark.asyncio
async def test_executor_prepends_global_rules(product_skill):
    """测试全局规则被拼接到 skill prompt 前面。"""
    registry = make_mock_registry(product_skill)
    mock_response = MagicMock()
    mock_response.content = "这款商品..."
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    state = AgentState(
        messages=[HumanMessage(content="介绍一下商品")],
        user_id="u1",
        intent="product",
        bargain_count=0,
        item_info={},
        manual_mode=False,
    )

    with patch("agents.nodes.skill_executor.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor.check_safety", side_effect=lambda x: x), \
         patch("agents.nodes.skill_executor._GLOBAL_RULES_PATH") as mock_rules_path:
        mock_rules_path.read_text.return_value = "全局规则内容"
        executor = make_skill_executor(registry)
    await executor(state)

    call_args = mock_llm.invoke.call_args[0][0]
    system_msg = call_args[0]
    assert isinstance(system_msg, SystemMessage)
    assert system_msg.content.startswith("全局规则内容\n\n")
    assert product_skill.prompt in system_msg.content


@pytest.mark.asyncio
async def test_executor_missing_rules_file_continues(product_skill):
    """测试全局规则文件缺失时优雅降级。"""
    registry = make_mock_registry(product_skill)
    mock_response = MagicMock()
    mock_response.content = "这款商品..."
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    state = AgentState(
        messages=[HumanMessage(content="介绍一下商品")],
        user_id="u1",
        intent="product",
        bargain_count=0,
        item_info={},
        manual_mode=False,
    )

    with patch("agents.nodes.skill_executor.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor.check_safety", side_effect=lambda x: x), \
         patch("agents.nodes.skill_executor._GLOBAL_RULES_PATH") as mock_rules_path:
        mock_rules_path.read_text.side_effect = FileNotFoundError
        executor = make_skill_executor(registry)
    await executor(state)

    assert mock_llm.invoke.called
    call_args = mock_llm.invoke.call_args[0][0]
    system_msg = call_args[0]
    assert isinstance(system_msg, SystemMessage)
    assert system_msg.content == product_skill.prompt  # no prefix
