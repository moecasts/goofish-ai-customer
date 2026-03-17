"""测试 Agent 状态图。"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from agents.graph import create_agent_graph, route_intent, LangGraphRouter
from agents.state import AgentState


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """创建临时 skills 目录供 graph 测试使用。"""
    for name, desc in [("price", "议价"), ("product", "商品咨询"), ("default", "默认回复")]:
        d = tmp_path / name
        d.mkdir()
        hooks = "\nstate_hooks:\n  - bargain_count\n  - min_price" if name == "price" else ""
        (d / "skill.md").write_text(
            f"---\nname: {name}\ndescription: {desc}{hooks}\n---\n\n{name} prompt"
        )
    return tmp_path


def test_create_agent_graph(skills_dir):
    """测试状态图可以成功创建。"""
    graph = create_agent_graph(str(skills_dir))
    assert graph is not None


def test_graph_structure(skills_dir):
    """测试状态图包含必要节点。"""
    graph = create_agent_graph(str(skills_dir))
    graph_nodes = set(graph.get_graph().nodes.keys())
    assert "classify" in graph_nodes
    assert "skill_executor" in graph_nodes


def test_route_intent_no_reply():
    """测试 no_reply 路由到 no_reply。"""
    state = AgentState(messages=[], user_id="u", intent="no_reply",
                       bargain_count=0, item_info=None, manual_mode=False)
    assert route_intent(state) == "no_reply"


def test_route_intent_price():
    """测试 price intent 路由到 skill。"""
    state = AgentState(messages=[], user_id="u", intent="price",
                       bargain_count=0, item_info=None, manual_mode=False)
    assert route_intent(state) == "skill"


def test_route_intent_default():
    """测试 default intent 路由到 skill。"""
    state = AgentState(messages=[], user_id="u", intent="default",
                       bargain_count=0, item_info=None, manual_mode=False)
    assert route_intent(state) == "skill"


def test_route_intent_missing():
    """测试 intent 为空时路由到 skill（默认 default）。"""
    state = AgentState(messages=[], user_id="u", intent="",
                       bargain_count=0, item_info=None, manual_mode=False)
    assert route_intent(state) == "skill"


@pytest.mark.asyncio
async def test_graph_executes_end_to_end(skills_dir):
    """测试完整图执行流程：classify → skill_executor → AIMessage 结果。"""
    mock_llm = AsyncMock()

    with patch("agents.nodes.classify.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor.check_safety", side_effect=lambda x: x):
        graph = create_agent_graph(str(skills_dir))
        state = {
            "messages": [HumanMessage(content="能便宜一点吗")],
            "user_id": "test_user",
            "bargain_count": 0,
            "item_info": {"min_price": "100"},
            "intent": "",
            "manual_mode": False,
        }
        # Mock classify to return "default" so we don't depend on LLM returning a valid skill name
        mock_llm.invoke.side_effect = [
            MagicMock(content="default"),   # classify call → returns "default"
            MagicMock(content="测试回复"),  # skill_executor call → returns response
        ]
        result = await graph.ainvoke(state, {"configurable": {"thread_id": "test_e2e"}})

    assert result.get("messages")
    last_msg = result["messages"][-1]
    assert isinstance(last_msg, AIMessage)
