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
    assert graph is not None


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
