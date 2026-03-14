"""商品咨询节点。"""

from langchain_core.messages import AIMessage, SystemMessage
from agents.state import AgentState
from services.llm_client import LLMClient
from services.tools import check_safety
from loguru import logger


async def product_node(state: AgentState) -> AgentState:
    """商品咨询节点。"""
    llm_client = LLMClient()

    try:
        with open("config/prompts/product_prompt.md", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        logger.error("product_prompt.md 文件未找到")
        return {"messages": [AIMessage(content="抱歉，暂时无法处理商品咨询")]}

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"],
    ]

    try:
        response = await llm_client.invoke(messages, allow_empty=False)
        logger.debug(f"LLM 原始回复: {response.content}")
        safe_content = check_safety(response.content)
        logger.debug(f"安全过滤后: {safe_content}")
        logger.info(f"商品咨询节点执行成功, reply_length={len(safe_content)}")
        return {"messages": [AIMessage(content=safe_content)]}
    except Exception as e:
        logger.error(f"商品咨询节点执行失败: {e}")
        return {"messages": [AIMessage(content="抱歉，处理商品咨询时出错了")]}
