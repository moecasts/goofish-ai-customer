"""测试意图识别节点。"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from agents.nodes.classify import make_classify_node
from agents.skill_registry import Skill, SkillRegistry
from agents.state import AgentState
from langchain_core.messages import HumanMessage


def make_registry_with_skills(*names: str) -> MagicMock:
    """创建包含指定 skill names 的 mock registry。"""
    registry = MagicMock(spec=SkillRegistry)
    skills = [Skill(name=n, description=f"{n} 描述", prompt="", skill_dir=MagicMock()) for n in names]
    registry.list_skills.return_value = skills
    registry.build_classify_context.return_value = "\n".join(
        f"- {s.name}: {s.description}" for s in skills
    )
    return registry


@pytest.mark.asyncio
async def test_classify_node_valid_intent():
    """测试识别有效意图。"""
    registry = make_registry_with_skills("price", "product", "default")
    mock_response = MagicMock()
    mock_response.content = "price"
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    with patch("agents.nodes.classify.LLMClient", return_value=mock_llm), \
         patch("builtins.open", mock_open(read_data="分类 prompt\n{skills}")):
        node = make_classify_node(registry)
        state = AgentState(messages=[HumanMessage(content="能便宜点吗")])
        result = await node(state)

    assert result["intent"] == "price"


@pytest.mark.asyncio
async def test_classify_node_invalid_intent_falls_back_to_default():
    """测试未知 intent 降级为 default。"""
    registry = make_registry_with_skills("price", "product", "default")
    mock_response = MagicMock()
    mock_response.content = "totally_unknown"
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    with patch("agents.nodes.classify.LLMClient", return_value=mock_llm), \
         patch("builtins.open", mock_open(read_data="分类 prompt\n{skills}")):
        node = make_classify_node(registry)
        state = AgentState(messages=[HumanMessage(content="随便")])
        result = await node(state)

    assert result["intent"] == "default"


@pytest.mark.asyncio
async def test_classify_node_no_reply_is_valid():
    """测试 no_reply 是合法 intent（内置保留值）。"""
    registry = make_registry_with_skills("price", "product", "default")
    mock_response = MagicMock()
    mock_response.content = "no_reply"
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    with patch("agents.nodes.classify.LLMClient", return_value=mock_llm), \
         patch("builtins.open", mock_open(read_data="分类 prompt\n{skills}")):
        node = make_classify_node(registry)
        state = AgentState(messages=[HumanMessage(content="😊")])
        result = await node(state)

    assert result["intent"] == "no_reply"


@pytest.mark.asyncio
async def test_classify_node_file_not_found():
    """测试 prompt 文件不存在时降级为 default。"""
    registry = make_registry_with_skills("price", "product", "default")
    with patch("builtins.open", side_effect=FileNotFoundError):
        node = make_classify_node(registry)
        state = AgentState(messages=[HumanMessage(content="你好")])
        result = await node(state)

    assert result["intent"] == "default"


@pytest.mark.asyncio
async def test_classify_node_llm_error():
    """测试 LLM 调用失败时降级为 default。"""
    registry = make_registry_with_skills("price", "product", "default")
    mock_llm = AsyncMock()
    mock_llm.invoke.side_effect = Exception("LLM 挂了")

    with patch("agents.nodes.classify.LLMClient", return_value=mock_llm), \
         patch("builtins.open", mock_open(read_data="分类 prompt\n{skills}")):
        node = make_classify_node(registry)
        state = AgentState(messages=[HumanMessage(content="你好")])
        result = await node(state)

    assert result["intent"] == "default"
