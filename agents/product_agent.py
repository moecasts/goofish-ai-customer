"""商品咨询 Agent。"""

from agents.default_agent import BaseAgent


class ProductAgent(BaseAgent):
    def __init__(self):
        super().__init__("config/prompts/product_prompt.md", temperature=0.4)
