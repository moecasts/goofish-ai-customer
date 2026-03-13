import pytest


@pytest.mark.unit
class TestIntentRouter:
    """Test suite for IntentRouter class."""

    @pytest.mark.parametrize(
        "query,expected_intent",
        [
            ("能便宜点吗", "price"),
            ("最低多少钱卖", "price"),
            ("500元可以吗", "price"),
            ("能少50吗", "price"),
            ("参数是什么", "product"),
            ("什么型号", "product"),
            ("和iPhone14比怎么样", "product"),
        ],
    )
    def test_keyword_match(self, intent_router, query, expected_intent):
        """Test keyword matching for various user queries."""
        assert intent_router.keyword_match(query) == expected_intent

    def test_product_priority_over_price(self, intent_router):
        """Product keywords should have priority over price keywords."""
        # "这个规格多少钱" contains both product and price keywords, product takes priority
        assert intent_router.keyword_match("这个规格多少钱") == "product"

    @pytest.mark.parametrize("query", ["你好", "在吗", "谢谢"])
    def test_no_keyword_match(self, intent_router, query):
        """Test queries that should not match any keywords."""
        assert intent_router.keyword_match(query) is None


@pytest.mark.unit
class TestSafetyFilter:
    """Test suite for safety filter functionality."""

    def test_blocks_wechat(self):
        from agents.default_agent import BaseAgent

        assert "安全提醒" in BaseAgent.safe_filter("加我微信吧")

    def test_blocks_qq(self):
        from agents.default_agent import BaseAgent

        assert "安全提醒" in BaseAgent.safe_filter("QQ联系")

    def test_passes_normal(self):
        from agents.default_agent import BaseAgent

        assert BaseAgent.safe_filter("商品不错") == "商品不错"


@pytest.mark.unit
class TestPriceAgentTemperature:
    """Test suite for PriceAgent temperature calculations."""

    @pytest.mark.parametrize(
        "round,expected_temp",
        [
            (0, 0.3),
            (1, 0.45),
            (2, 0.6),
            (4, 0.9),
            (10, 0.9),  # capped at maximum
        ],
    )
    def test_dynamic_temperature(self, price_agent, round, expected_temp):
        """Test temperature calculation for different bargaining rounds."""
        assert price_agent.get_temperature(round) == pytest.approx(expected_temp)
