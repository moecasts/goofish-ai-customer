"""议价 Agent。动态 temperature 随议价次数递增。"""

from agents.default_agent import BaseAgent


class PriceAgent(BaseAgent):
    def __init__(self):
        super().__init__("config/prompts/price_prompt.md", temperature=0.3)

    @staticmethod
    def get_temperature(bargain_count: int) -> float:
        return min(0.3 + bargain_count * 0.15, 0.9)

    async def generate(self, user_msg: str, item_desc: str = "", context: str = "", **kwargs) -> str:
        bargain_count = kwargs.get("bargain_count", 0)
        self.temperature = self.get_temperature(bargain_count)
        return await super().generate(user_msg, item_desc, context, **kwargs)
