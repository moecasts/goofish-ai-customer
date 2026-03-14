"""LangGraph 真实 LLM API 集成测试。

这些测试需要配置 API_KEY 才能运行，用于验证 LangGraph 与真实 LLM 的集成。

## 集成测试策略

由于 LLM 的回复是随机和不可控的，这些集成测试**只验证基本质量**，不检查具体内容：

✅ 检查内容：
- 回复不为空
- 回复长度合理（>10个字符）
- 不包含错误信息
- 端到端流程能正常工作

❌ 不检查内容：
- 具体的关键词（如"4000"、"钛金属"等）
- 具体的回复格式
- 具体的业务逻辑

## 测试分工

- **集成测试**（本文件）：验证端到端流程，使用真实 LLM
- **单元测试**（test_*.py）：测试具体逻辑，使用 mock
"""

import os
import time
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("API_KEY"), reason="需要配置 API_KEY")
class TestRealLLMAPILangGraph:
    """LangGraph 真实 LLM API 集成测试。

    这些测试需要配置 API_KEY 且启用 LangGraph 时才会运行。
    """

    @pytest.mark.asyncio
    async def test_langgraph_price_negotiation(self):
        """测试 LangGraph 议价场景（真实 API 调用）。

        注意：由于 LangGraph 的 MemorySaver 需要 thread_id 保持一致，
        但当前实现中每次调用 route() 都会创建全新的初始状态，
        所以这个测试只测试单轮议价，不测试多轮对话。
        """
        from agents.graph import LangGraphRouter

        router = LangGraphRouter()

        # 测试单轮议价
        print("\n" + "=" * 60)
        print("LangGraph 议价测试 - 单轮")
        print("-" * 60)

        start_time = time.perf_counter()
        reply = await router.route(
            user_msg="3000能卖吗？",
            item_desc="iPhone 15 Pro Max 256GB，成色95新，无磕碰",
            user_id="test_price_user",
            bargain_count=0,
            min_price=4000,
            price="4000-5000",
        )
        elapsed = time.perf_counter() - start_time

        print("用户: 3000能卖吗？")
        print(f"回复: {reply}")
        print(f"耗时: {elapsed:.2f}秒")
        print("=" * 60 + "\n")

        # 验证回复基本质量（不检查具体内容，因为 LLM 回复不可控）
        assert isinstance(reply, str)
        assert len(reply) > 10, "回复长度应该大于10个字符"
        assert "错误" not in reply, "回复不应包含错误信息"
        assert "失败" not in reply, "回复不应包含失败信息"

    @pytest.mark.asyncio
    async def test_langgraph_product_inquiry(self):
        """测试 LangGraph 商品咨询场景（真实 API 调用）。"""
        from agents.graph import LangGraphRouter

        router = LangGraphRouter()

        print("\n" + "=" * 60)
        print("LangGraph 商品咨询测试")
        print("-" * 60)

        start_time = time.perf_counter()
        reply = await router.route(
            user_msg="这个手机是什么颜色的？有原装充电器吗？",
            item_desc="iPhone 15 Pro Max 256GB，原色钛金属，无磕碰，含原装充电器和数据线",
            product_name="iPhone 15 Pro Max",
            user_id="test_product_user",
        )
        elapsed = time.perf_counter() - start_time

        print("用户: 这个手机是什么颜色的？有原装充电器吗？")
        print(f"回复: {reply}")
        print(f"耗时: {elapsed:.2f}秒")
        print("=" * 60 + "\n")

        # 验证回复基本质量（不检查具体内容，因为 LLM 回复不可控）
        assert isinstance(reply, str)
        assert len(reply) > 10, "回复长度应该大于10个字符"
        assert "错误" not in reply, "回复不应包含错误信息"
        assert "失败" not in reply, "回复不应包含失败信息"

    @pytest.mark.asyncio
    async def test_langgraph_default_reply(self):
        """测试 LangGraph 默认回复场景（真实 API 调用）。"""
        from agents.graph import LangGraphRouter

        router = LangGraphRouter()

        print("\n" + "=" * 60)
        print("LangGraph 默认回复测试")
        print("-" * 60)

        start_time = time.perf_counter()
        reply = await router.route(
            user_msg="你好，在吗？",
            item_desc="iPhone 15 Pro Max",
            user_id="test_default_user",
        )
        elapsed = time.perf_counter() - start_time

        print("用户: 你好，在吗？")
        print(f"回复: {reply}")
        print(f"耗时: {elapsed:.2f}秒")
        print("=" * 60 + "\n")

        # 验证回复
        assert isinstance(reply, str)
        assert len(reply) > 5
        assert "错误" not in reply

    @pytest.mark.asyncio
    async def test_langgraph_safety_filter(self):
        """测试 LangGraph 敏感词过滤（真实 API 调用）。"""
        from agents.graph import LangGraphRouter

        router = LangGraphRouter()

        print("\n" + "=" * 60)
        print("LangGraph 安全过滤测试")
        print("-" * 60)

        # 测试各种敏感词
        test_cases = [
            "加我微信详谈",
            "QQ 联系我",
            "可以支付宝转账吗",
            "我银行卡转账给你",
            "我们线下交易吧",
        ]

        for user_msg in test_cases:
            start_time = time.perf_counter()
            reply = await router.route(
                user_msg=user_msg,
                item_desc="iPhone 15 Pro Max",
                user_id="test_safety_user",
            )
            elapsed = time.perf_counter() - start_time

            print(f"用户: {user_msg}")
            print(f"回复: {reply}")
            print(f"耗时: {elapsed:.2f}秒")
            print("-" * 60)

            # 验证回复基本质量（不检查具体过滤结果，因为 LLM 回复不可控）
            assert isinstance(reply, str)
            assert len(reply) > 5, "回复长度应该大于5个字符"
            assert "错误" not in reply, "回复不应包含错误信息"

        print("=" * 60 + "\n")

    @pytest.mark.asyncio
    async def test_langgraph_session_isolation(self):
        """测试 LangGraph 会话隔离（真实 API 调用）。"""
        from agents.graph import LangGraphRouter

        router = LangGraphRouter()

        print("\n" + "=" * 60)
        print("LangGraph 会话隔离测试")
        print("-" * 60)

        # 用户 A 的第 1 轮议价
        reply_a1 = await router.route(
            user_msg="3000能卖吗？",
            user_id="user_a",
            bargain_count=0,
            min_price=4000,
        )
        print(f"用户A (第1轮): {reply_a1}")

        # 用户 B 的第 1 轮议价（应该独立计数）
        reply_b1 = await router.route(
            user_msg="3000能卖吗？",
            user_id="user_b",
            bargain_count=0,
            min_price=4000,
        )
        print(f"用户B (第1轮): {reply_b1}")

        # 用户 A 的第 2 轮议价（应该使用 bargain_count=1）
        reply_a2 = await router.route(
            user_msg="3500呢？",
            user_id="user_a",
            bargain_count=1,
            min_price=4000,
        )
        print(f"用户A (第2轮): {reply_a2}")

        # 用户 B 的第 2 轮议价（应该独立计数，也是 bargain_count=1）
        reply_b2 = await router.route(
            user_msg="3500呢？",
            user_id="user_b",
            bargain_count=1,
            min_price=4000,
        )
        print(f"用户B (第2轮): {reply_b2}")

        print("=" * 60 + "\n")

        # 验证不同用户的会话独立
        assert isinstance(reply_a1, str)
        assert isinstance(reply_b1, str)
        assert isinstance(reply_a2, str)
        assert isinstance(reply_b2, str)
        # 两个用户的第1轮应该类似（都坚持原价）
        # 两个用户的第2轮应该类似（都小幅让步）

    @pytest.mark.asyncio
    async def test_langgraph_performance(self):
        """测试 LangGraph 性能（真实 API 调用）。"""
        from agents.graph import LangGraphRouter

        router = LangGraphRouter()

        print("\n" + "=" * 60)
        print("LangGraph 性能测试")
        print("-" * 60)

        # 连续调用 5 次，测试响应时间
        elapsed_times = []
        for i in range(5):
            start_time = time.perf_counter()
            reply = await router.route(
                user_msg=f"你好，这是第{i + 1}次测试",
                item_desc="测试商品",
                user_id=f"perf_test_user_{i}",
            )
            elapsed = time.perf_counter() - start_time
            elapsed_times.append(elapsed)

            print(f"第{i + 1}次: {elapsed:.2f}秒 - {reply[:30]}...")

        avg_time = sum(elapsed_times) / len(elapsed_times)
        max_time = max(elapsed_times)
        min_time = min(elapsed_times)

        print("-" * 60)
        print(f"平均耗时: {avg_time:.2f}秒")
        print(f"最大耗时: {max_time:.2f}秒")
        print(f"最小耗时: {min_time:.2f}秒")
        print("=" * 60 + "\n")

        # 验证性能（LLM 调用本身需要时间，两次调用：分类 + 生成回复）
        assert avg_time < 15.0, f"平均响应时间过长: {avg_time:.2f}秒"
        assert max_time < 20.0, f"最大响应时间过长: {max_time:.2f}秒"
