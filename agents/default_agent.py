"""默认回复 Agent + BaseAgent 基类。"""

import os
from openai import AsyncOpenAI
from loguru import logger


class LLMError(Exception):
    """LLM 接口调用失败。"""

    pass


class BaseAgent:
    BLOCKED_PHRASES = ["微信", "QQ", "支付宝", "银行卡", "线下"]

    def __init__(self, prompt_path: str, temperature: float = 0.7):
        self.prompt_path = prompt_path
        self.temperature = temperature
        self.system_prompt = self._load_prompt()

        api_key = os.getenv("API_KEY")
        if api_key:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=os.getenv(
                    "MODEL_BASE_URL",
                    "https://open.bigmodel.cn/api/paas/v4/",
                ),
            )
            self.model = os.getenv("MODEL_NAME", "glm-4.7-flash")
        else:
            self.client = None
            self.model = None
            logger.warning(
                "API_KEY not configured. LLM features will be disabled. Please set API_KEY in .env file to enable LLM functionality."
            )

    def _load_prompt(self) -> str:
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {self.prompt_path}")
            return ""

    @staticmethod
    def safe_filter(text: str) -> str:
        for phrase in BaseAgent.BLOCKED_PHRASES:
            if phrase in text:
                return "[安全提醒] 请通过闲鱼平台沟通，不要在站外交易哦"
        return text

    async def generate(
        self, user_msg: str, item_desc: str = "", context: str = "", **kwargs
    ) -> str:
        # 如果 client 不可用，返回默认提示
        if not self.client:
            return "抱歉，AI 功能暂时不可用，请稍后再试。"

        prompt = self.system_prompt
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        messages = [
            {
                "role": "system",
                "content": f"【商品信息】{item_desc}\n【对话历史】{context}\n{prompt}",
            },
            {"role": "user", "content": user_msg},
        ]

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=500,
                top_p=0.8,
            )
            reply = resp.choices[0].message.content.strip()
            return self.safe_filter(reply)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise LLMError(f"LLM call failed: {e}") from e


class DefaultAgent(BaseAgent):
    def __init__(self):
        super().__init__("config/prompts/default_prompt.md", temperature=0.7)
