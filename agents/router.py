"""三级意图路由：关键词匹配 -> LLM 分类 -> Agent 分发。"""

import os
import re
from loguru import logger
from agents.classify_agent import ClassifyAgent
from agents.price_agent import PriceAgent
from agents.product_agent import ProductAgent
from agents.default_agent import DefaultAgent, LLMError


class IntentRouter:
    FALLBACK_REPLY = os.getenv("FALLBACK_REPLY", "卖家暂时离开了，回来马上回复！")
    PRODUCT_KEYWORDS = ["参数", "规格", "型号", "连接", "对比"]
    PRODUCT_PATTERNS = [re.compile(r"和.+比")]

    PRICE_KEYWORDS = ["便宜", "价", "砍价", "少点", "多少钱", "最低"]
    PRICE_PATTERNS = [re.compile(r"\d+元"), re.compile(r"能少\d+")]

    def __init__(self):
        self.classify_agent = ClassifyAgent()
        self.agents = {
            "price": PriceAgent(),
            "product": ProductAgent(),
            "default": DefaultAgent(),
        }

    def keyword_match(self, text: str) -> str | None:
        # 商品关键词优先
        for kw in self.PRODUCT_KEYWORDS:
            if kw in text:
                return "product"
        for pat in self.PRODUCT_PATTERNS:
            if pat.search(text):
                return "product"

        # 议价关键词
        for kw in self.PRICE_KEYWORDS:
            if kw in text:
                return "price"
        for pat in self.PRICE_PATTERNS:
            if pat.search(text):
                return "price"

        return None

    async def route(
        self, user_msg: str, item_desc: str = "", context: str = "", **kwargs
    ) -> str:
        # 一级：关键词匹配
        intent = self.keyword_match(user_msg)
        if intent:
            logger.info(f"Keyword match: {intent}")
        else:
            # 二级：LLM 分类
            intent = await self.classify_agent.classify(user_msg, item_desc, context)
            logger.info(f"LLM classify: {intent}")

        if intent == "no_reply":
            return ""

        # 三级：Agent 生成回复
        agent = self.agents.get(intent, self.agents["default"])
        try:
            return await agent.generate(user_msg, item_desc, context, **kwargs)
        except LLMError:
            logger.warning(f"Agent '{intent}' generation failed, using fallback reply")
            return self.FALLBACK_REPLY
