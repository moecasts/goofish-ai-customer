import os
import time

import pytest
from dotenv import load_dotenv
from pprint import pprint

# Load environment variables from .env file
load_dotenv()


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


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("API_KEY"), reason="API_KEY not set in .env")
class TestRealLLMAPI:
    """Integration tests for real LLM API calls.

    These tests only run when API_KEY is configured in .env file.
    They make actual API calls to verify the integration works correctly.
    """

    @pytest.mark.asyncio
    async def test_default_agent_real_api(self):
        """Test DefaultAgent with real LLM API call."""
        from agents.default_agent import DefaultAgent

        # Create real agent instance (not mocked)
        agent = DefaultAgent()

        # Verify agent is properly configured
        assert agent.client is not None, (
            "Agent client should be configured when API_KEY is set"
        )
        assert agent.model is not None, (
            "Agent model should be configured when API_KEY is set"
        )

        # Make a real API call
        user_msg = "你好，请问这个商品还在吗？"
        item_desc = "测试商品"

        start_time = time.perf_counter()
        response = await agent.generate(
            user_msg=user_msg,
            item_desc=item_desc,
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Output user message and response for debugging
        print("\n" + "=" * 60)
        print("DefaultAgent Test:")
        print(f"User Message: {user_msg}")
        print(f"Item Description: {item_desc}")
        print(f"API Call Time: {elapsed_time:.2f} seconds")
        print("-" * 60)
        print("Response:")
        pprint(response, width=80)
        print("=" * 60 + "\n")

        # Verify response is valid
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 10, "Response should not be too short"
        assert len(response) < 1000, "Response should not be too long"
        assert "错误" not in response, "Response should not contain error messages"
        assert "失败" not in response, "Response should not contain failure messages"
        assert "error" not in response.lower(), "Response should not contain 'error'"

    @pytest.mark.asyncio
    async def test_price_agent_real_api(self):
        """Test PriceAgent with real LLM API call."""
        from agents.price_agent import PriceAgent

        # Create real agent instance (not mocked)
        agent = PriceAgent()

        # Verify agent is properly configured
        assert agent.client is not None, (
            "Agent client should be configured when API_KEY is set"
        )
        assert agent.model is not None, (
            "Agent model should be configured when API_KEY is set"
        )

        # Make a real API call with price-related query
        user_msg = "能便宜点吗？"
        item_desc = "二手手机，成色新"
        bargain_count = 1

        start_time = time.perf_counter()
        response = await agent.generate(
            user_msg=user_msg,
            item_desc=item_desc,
            bargain_count=bargain_count,
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Output user message and response for debugging
        print("\n" + "=" * 60)
        print("PriceAgent Test:")
        print(f"User Message: {user_msg}")
        print(f"Item Description: {item_desc}")
        print(f"Bargain Count: {bargain_count}")
        print(f"API Call Time: {elapsed_time:.2f} seconds")
        print("-" * 60)
        print("Response:")
        pprint(response, width=80)
        print("=" * 60 + "\n")

        # Verify response is valid
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 10, "Response should not be too short"
        assert len(response) < 1000, "Response should not be too long"
        assert "错误" not in response, "Response should not contain error messages"
        assert "失败" not in response, "Response should not contain failure messages"
        assert "error" not in response.lower(), "Response should not contain 'error'"
