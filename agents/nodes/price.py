"""议价节点。"""

from langchain_core.messages import AIMessage, SystemMessage
from agents.state import AgentState
from services.llm_client import LLMClient
from services.tools import check_safety
from loguru import logger


async def price_node(state: AgentState) -> AgentState:
    """议价节点。"""
    llm_client = LLMClient()

    try:
        with open("config/prompts/price_prompt.md", "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        logger.error("price_prompt.md 文件未找到")
        return {"messages": [AIMessage(content="抱歉，暂时无法处理议价请求")]}

    item_info = state["item_info"] or {}
    prompt = prompt_template.replace("{min_price}", str(item_info.get("min_price", "")))
    prompt = prompt.replace("{bargain_count}", str(state["bargain_count"]))

    messages = [
        SystemMessage(content=prompt),
        *state["messages"],
    ]

    try:
        temperature = min(0.3 + state["bargain_count"] * 0.15, 0.9)
        response = await llm_client.invoke(
            messages, temperature=temperature, allow_empty=False
        )
        logger.debug(f"LLM 原始回复: {response.content}")
        safe_content = check_safety(response.content)
        logger.debug(f"安全过滤后: {safe_content}")
        logger.info(
            f"议价节点: count={state['bargain_count']}, temp={temperature}, reply_length={len(safe_content)}"
        )
        return {
            "messages": [AIMessage(content=safe_content)],
            "bargain_count": state["bargain_count"] + 1,
        }
    except Exception as e:
        logger.error(f"议价节点执行失败: {e}")
        return {
            "messages": [AIMessage(content="抱歉，处理议价请求时出错了")],
            "bargain_count": state["bargain_count"],
        }


def calculate_price_temperature(bargain_count: int) -> float:
    """计算议价温度。"""
    return min(0.3 + bargain_count * 0.15, 0.9)
