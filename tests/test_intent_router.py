"""测试意图路由函数。"""

import pytest
from agents.graph import route_intent
from agents.state import AgentState


def make_state(intent: str) -> AgentState:
    return AgentState(messages=[], user_id="u", intent=intent,
                      bargain_count=0, item_info=None, manual_mode=False)


def test_route_intent_price():
    assert route_intent(make_state("price")) == "skill"

def test_route_intent_product():
    assert route_intent(make_state("product")) == "skill"

def test_route_intent_default():
    assert route_intent(make_state("default")) == "skill"

def test_route_intent_no_reply():
    assert route_intent(make_state("no_reply")) == "no_reply"

def test_route_intent_missing():
    assert route_intent(make_state("")) == "skill"
