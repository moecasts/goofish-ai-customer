"""LangGraph 路由器测试。"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.graph import LangGraphRouter


class TestLangGraphRouter:
    """测试 LangGraph 路由器。"""

    @pytest.fixture
    def mock_graph(self):
        """模拟 LangGraph。"""
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [
                MagicMock(content="你好！这是一条测试回复。")
            ]
        }
        # get_state is called synchronously; return None so existing_state defaults to {}
        mock_graph.get_state = MagicMock(return_value=None)
        return mock_graph

    @pytest.fixture
    def router(self, mock_graph):
        """创建路由器实例。"""
        with patch('agents.graph.create_agent_graph', return_value=mock_graph):
            return LangGraphRouter()

    @pytest.mark.asyncio
    async def test_route_success(self, router, mock_graph):
        """测试成功路由。"""
        from langchain_core.messages import AIMessage

        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="你好！这是一条测试回复。")]
        }

        result = await router.route(
            user_msg="你好，这个商品还在吗？",
            item_desc="iPhone 15 Pro 128GB",
            user_id="test_user",
            bargain_count=1
        )

        assert result[0] == "你好！这是一条测试回复。"
        mock_graph.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_no_messages(self, router, mock_graph):
        """测试没有消息的情况。"""
        mock_graph.ainvoke.return_value = {"messages": []}

        result = await router.route(user_msg="你好")

        assert result[0] == ""

    @pytest.mark.asyncio
    async def test_route_exception(self, router, mock_graph):
        """测试异常处理。"""
        mock_graph.ainvoke.side_effect = Exception("执行失败")

        result = await router.route(user_msg="你好")

        assert result[0] == "卖家暂时离开了，回来马上回复！"

    @pytest.mark.asyncio
    async def test_route_with_context(self, router, mock_graph):
        """测试带上下文的路由。"""
        from langchain_core.messages import AIMessage

        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="好的，我知道了。")]
        }

        result = await router.route(
            user_msg="多少钱？",
            context="用户之前问了库存问题"
        )

        assert result[0] == "好的，我知道了。"

    @pytest.mark.asyncio
    async def test_route_with_item_info(self, router, mock_graph):
        """测试带商品信息的路由。"""
        from langchain_core.messages import AIMessage

        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="这个价格不贵，成色很好。")]
        }

        result = await router.route(
            user_msg="能便宜点吗？",
            item_desc="iPhone 15 Pro 256GB，成色新",
            min_price="4000",
            price="4000-5000",
            bargain_count=1,
        )

        assert result[0] == "这个价格不贵，成色很好。"

    @pytest.mark.asyncio
    async def test_session_isolation(self, router, mock_graph):
        """测试会话隔离。"""
        from langchain_core.messages import AIMessage

        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="回复内容")]
        }

        # 不同用户ID
        await router.route(user_msg="测试", user_id="user_a")
        await router.route(user_msg="测试", user_id="user_b")

        # 验证调用了两次（会话隔离）
        assert mock_graph.ainvoke.call_count == 2

        # 验证每次调用都传入了状态参数
        for call in mock_graph.ainvoke.call_args_list:
            args = call.args
            assert len(args) > 0, "ainvoke should be called with state argument"
            state = args[0]
            assert "messages" in state, "State should contain messages"
