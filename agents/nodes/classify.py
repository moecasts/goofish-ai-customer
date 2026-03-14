"""意图识别节点。"""

from langchain_core.messages import SystemMessage
from agents.state import AgentState
from services.llm_client import LLMClient
from loguru import logger


async def classify_node(state: AgentState) -> AgentState:
    """意图识别节点。"""
    llm_client = LLMClient()

    try:
        with open("config/prompts/classify_prompt.md", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        logger.error("classify_prompt.md 文件未找到")
        return {"intent": "default"}

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"],
    ]

    try:
        response = await llm_client.invoke(messages, temperature=0.3)
        intent = response.content.strip().lower()
        valid_intents = {"price", "product", "default", "no_reply"}
        intent = intent if intent in valid_intents else "default"
        logger.info(f"意图识别结果: {intent}")
        return {"intent": intent}
    except Exception as e:
        logger.error(f"意图识别失败: {e}")
        return {"intent": "default"}
