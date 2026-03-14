"""Agent 状态图构建。"""

import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from agents.state import AgentState
from agents.nodes import classify_node, price_node, product_node, default_node
from services.llm_client import LLMClient


def route_intent(state: AgentState) -> str:
    """根据意图路由到不同节点。"""
    intent = state.get("intent", "default") or "default"
    logger.info(f"路由意图: {state.get('intent')} -> {intent}")

    if intent == "no_reply":
        return "no_reply"

    return intent


def check_bargain_continue(state: AgentState) -> str:
    """检查是否继续议价。

    注意：每次调用都会生成一次回复，然后结束。
    下一轮议价需要用户新发送消息触发。
    """
    # 简化逻辑：每次只回复一次，不自动循环
    # 用户想继续议价时会发送新消息
    logger.info(
        f"议价回复已生成 (bargain_count={state['bargain_count']})，等待用户下一条消息"
    )
    return "stop"


def create_agent_graph():
    """创建 Agent 状态图。"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("classify", classify_node)
    workflow.add_node("price", price_node)
    workflow.add_node("product", product_node)
    workflow.add_node("default", default_node)

    # 设置入口点
    workflow.set_entry_point("classify")

    # 添加意图路由条件边
    workflow.add_conditional_edges(
        "classify",
        route_intent,
        {
            "price": "price",
            "product": "product",
            "default": "default",
            "no_reply": END,
        },
    )

    # 议价节点添加循环边
    workflow.add_conditional_edges(
        "price",
        check_bargain_continue,
        {
            "continue": "price",
            "stop": END,
        },
    )

    # 其他节点直接结束
    workflow.add_edge("product", END)
    workflow.add_edge("default", END)

    logger.info("Agent 状态图构建完成")

    # 配置 checkpointer（使用内存存储）
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


class LangGraphRouter:
    """LangGraph 路由器。"""

    def __init__(self):
        """初始化路由器。"""
        self.graph = create_agent_graph()
        self.llm_client = LLMClient()
        logger.info("LangGraphRouter 初始化完成")

    async def route(
        self,
        user_msg: str,
        item_desc: str = "",
        context: str = "",
        bargain_count: int = 0,
        min_price: str = "",
        product_name: str = "",
        price: str = "",
        description: str = "",
        user_id: str = "default_user",
        **kwargs: Any,
    ) -> tuple[str, int]:
        """路由到合适的 Agent 生成回复。

        Args:
            user_msg: 用户消息
            item_desc: 商品描述
            context: 对话上下文
            bargain_count: 议价次数（仅用于新会话的初始值）
            min_price: 最低价格
            product_name: 商品名称
            price: 价格范围
            description: 商品详情
            user_id: 用户 ID（用于会话隔离）
            **kwargs: 其他参数

        Returns:
            (Agent 生成的回复, 更新后的 bargain_count)
        """
        try:
            # 配置会话隔离
            config = {"configurable": {"thread_id": f"user_{user_id}"}}

            # 获取当前 checkpointed state（新会话时为空）
            current_state = self.graph.get_state(config)
            existing_state = current_state.values if current_state else {}

            # 构建更新状态，保留 checkpointed 值
            # 设计决策：只使用当前消息（price_prompt 要求"只回复用户最后一条消息，忽略之前的对话历史"）
            update_state: AgentState = {
                "messages": [HumanMessage(content=user_msg)],  # 当前消息
                "user_id": user_id,
                "bargain_count": existing_state.get(
                    "bargain_count", bargain_count
                ),  # 优先使用 checkpointed 值
                "item_info": {
                    "min_price": min_price,
                    "product_name": product_name,
                    "price": price,
                    "description": description,
                    "item_desc": item_desc,
                    "context": context,
                },
                "intent": "",  # 重置以重新分类意图
                "manual_mode": existing_state.get(
                    "manual_mode", False
                ),  # 保留 manual_mode 状态
            }

            # 执行状态图（使用 checkpointed state 作为基础）
            result = await self.graph.ainvoke(update_state, config)

            # 提取 AI 回复和更新的 bargain_count
            updated_bargain_count = result.get(
                "bargain_count", existing_state.get("bargain_count", bargain_count)
            )

            if result.get("messages"):
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    return last_message.content, updated_bargain_count
                elif isinstance(last_message, dict) and "content" in last_message:
                    return last_message["content"], updated_bargain_count

            return "", updated_bargain_count

        except Exception as e:
            logger.error(f"LangGraph 执行失败: {e}")
            return os.getenv(
                "FALLBACK_REPLY", "卖家暂时离开了，回来马上回复！"
            ), bargain_count
