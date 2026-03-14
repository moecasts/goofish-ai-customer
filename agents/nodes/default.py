"""默认回复节点。"""

from langchain_core.messages import AIMessage, SystemMessage
from agents.state import AgentState
from services.llm_client import LLMClient
from services.tools import check_safety
from loguru import logger


async def default_node(state: AgentState) -> AgentState:
    """默认回复节点。"""
    llm_client = LLMClient()

    try:
        with open("config/prompts/default_prompt.md", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        logger.error("default_prompt.md 文件未找到")
        return {"messages": [AIMessage(content="您好，请问有什么可以帮您的？")]}

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"],
    ]

    try:
        response = await llm_client.invoke(messages, temperature=0.7, allow_empty=False)
        safe_content = check_safety(response.content)
        logger.info("默认节点执行成功")
        return {"messages": [AIMessage(content=safe_content)]}
    except Exception as e:
        logger.error(f"默认节点执行失败: {e}")
        return {"messages": [AIMessage(content="抱歉，我现在无法回复，请稍后再试")]}
