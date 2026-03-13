"""意图分类 Agent。"""

from loguru import logger
from agents.default_agent import BaseAgent, LLMError


class ClassifyAgent(BaseAgent):
    VALID_INTENTS = {"price", "product", "default", "no_reply"}

    def __init__(self):
        super().__init__("config/prompts/classify_prompt.md", temperature=0.3)

    async def classify(self, user_msg: str, item_desc: str = "", context: str = "") -> str:
        try:
            result = await self.generate(user_msg, item_desc, context)
            result = result.strip().lower()
            return result if result in self.VALID_INTENTS else "default"
        except LLMError:
            logger.warning("Classification LLM failed, falling back to 'default' intent")
            return "default"
