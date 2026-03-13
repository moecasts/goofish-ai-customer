import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.router import IntentRouter
from agents.classify_agent import ClassifyAgent
from agents.price_agent import PriceAgent
from agents.product_agent import ProductAgent
from agents.default_agent import DefaultAgent


class TestIntentRouter:
    def setup_method(self):
        self.router = IntentRouter()

    def test_price_keyword_match(self):
        assert self.router.keyword_match("能便宜点吗") == "price"
        assert self.router.keyword_match("最低多少钱卖") == "price"
        assert self.router.keyword_match("500元可以吗") == "price"
        assert self.router.keyword_match("能少50吗") == "price"

    def test_product_keyword_match(self):
        assert self.router.keyword_match("参数是什么") == "product"
        assert self.router.keyword_match("什么型号") == "product"
        assert self.router.keyword_match("和iPhone14比怎么样") == "product"

    def test_product_priority_over_price(self):
        # "这个规格多少钱" 同时含商品和议价关键词，商品优先
        assert self.router.keyword_match("这个规格多少钱") == "product"

    def test_no_keyword_match(self):
        assert self.router.keyword_match("你好") is None
        assert self.router.keyword_match("在吗") is None


class TestSafetyFilter:
    def test_blocks_wechat(self):
        from agents.default_agent import BaseAgent
        assert "安全提醒" in BaseAgent.safe_filter("加我微信吧")

    def test_blocks_qq(self):
        from agents.default_agent import BaseAgent
        assert "安全提醒" in BaseAgent.safe_filter("QQ联系")

    def test_passes_normal(self):
        from agents.default_agent import BaseAgent
        assert BaseAgent.safe_filter("商品不错") == "商品不错"


class TestPriceAgentTemperature:
    def test_dynamic_temperature(self):
        agent = PriceAgent()
        assert agent.get_temperature(0) == pytest.approx(0.3)
        assert agent.get_temperature(1) == pytest.approx(0.45)
        assert agent.get_temperature(2) == pytest.approx(0.6)
        assert agent.get_temperature(4) == pytest.approx(0.9)
        assert agent.get_temperature(10) == pytest.approx(0.9)  # capped
