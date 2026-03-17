"""意图识别节点。"""

from langchain_core.messages import SystemMessage
from agents.state import AgentState
from agents.skill_registry import SkillRegistry
from services.llm_client import LLMClient
from loguru import logger


def make_classify_node(registry: SkillRegistry):
    """工厂函数：创建绑定了 registry 的 classify 节点函数。"""

    async def classify_node(state: AgentState) -> AgentState:
        """意图识别节点：动态注入 skill 列表，LLM 返回 skill name。"""
        llm_client = LLMClient()

        try:
            with open("config/prompts/classify_prompt.md", "r", encoding="utf-8") as f:
                prompt_template = f.read()
        except FileNotFoundError:
            logger.error("classify_prompt.md 文件未找到")
            return {"intent": "default"}

        # 动态注入 skill 列表
        skills_context = registry.build_classify_context()
        system_prompt = prompt_template.replace("{skills}", skills_context)

        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]

        # 合法的 intent = 所有 skill name + no_reply
        valid_intents = {s.name for s in registry.list_skills()} | {"no_reply"}

        try:
            response = await llm_client.invoke(messages)
            intent = response.content.strip().lower()
            if intent not in valid_intents:
                logger.warning(f"classify 返回未知 intent '{intent}'，降级为 default")
                intent = "default"
            logger.info(f"意图识别结果: {intent}")
            return {"intent": intent}
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            return {"intent": "default"}

    return classify_node
