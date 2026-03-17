"""Agent 状态图构建。"""

import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from agents.state import AgentState
from agents.nodes import make_classify_node, make_skill_executor
from agents.skill_registry import SkillRegistry

_SKILLS_DIR_DEFAULT = "config/skills"


def route_intent(state: AgentState) -> str:
    """no_reply 直接结束，其他全部走 skill_executor。"""
    intent = state.get("intent", "default") or "default"
    logger.info(f"路由意图: {intent}")
    if intent == "no_reply":
        return "no_reply"
    return "skill"


def create_agent_graph(skills_dir: str = _SKILLS_DIR_DEFAULT):
    """创建 Agent 状态图。"""
    registry = SkillRegistry(Path(skills_dir))
    logger.info(f"已加载 {len(registry.list_skills())} 个 skill: {[s.name for s in registry.list_skills()]}")

    workflow = StateGraph(AgentState)

    workflow.add_node("classify", make_classify_node(registry))
    workflow.add_node("skill_executor", make_skill_executor(registry))

    workflow.set_entry_point("classify")

    workflow.add_conditional_edges(
        "classify",
        route_intent,
        {
            "no_reply": END,
            "skill": "skill_executor",
        },
    )

    workflow.add_edge("skill_executor", END)

    logger.info("Agent 状态图构建完成")
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


class LangGraphRouter:
    """LangGraph 路由器。"""

    def __init__(self, skills_dir: str = _SKILLS_DIR_DEFAULT):
        """初始化路由器。"""
        self.graph = create_agent_graph(skills_dir)
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
            update_state: AgentState = {
                "messages": [HumanMessage(content=user_msg)],
                "user_id": user_id,
                "bargain_count": existing_state.get("bargain_count", bargain_count),
                "item_info": {
                    "min_price": min_price,
                    "product_name": product_name,
                    "price": price,
                    "description": description,
                    "item_desc": item_desc,
                    "context": context,
                },
                "intent": "",
                "manual_mode": existing_state.get("manual_mode", False),
            }

            # 执行状态图
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
