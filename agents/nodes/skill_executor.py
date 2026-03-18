"""通用 skill 执行节点。"""

import re
from pathlib import Path
from typing import Callable
from langchain_core.messages import AIMessage, SystemMessage
from agents.state import AgentState
from agents.skill_registry import SkillRegistry
from services.llm_client import LLMClient
from services.tools import check_safety
from loguru import logger

_GLOBAL_RULES_PATH = Path(__file__).parent.parent.parent / "config" / "rules" / "global.md"


def make_skill_executor(registry: SkillRegistry) -> Callable:
    """工厂函数：创建绑定了 registry 的 skill_executor 节点函数。"""
    llm_client = LLMClient()

    try:
        global_rules = _GLOBAL_RULES_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(f"全局规则文件未找到: {_GLOBAL_RULES_PATH}，跳过全局规则注入")
        global_rules = ""

    async def skill_executor_node(state: AgentState) -> AgentState:
        """通用 skill 执行节点：查找 skill → 注入变量 → 调用 LLM → 安全过滤 → 返回。"""
        intent = state.get("intent", "default") or "default"
        skill = registry.get_skill(intent)

        if skill is None:
            logger.warning(f"未找到 skill: {intent}，返回 fallback 回复")
            return {"messages": [AIMessage(content="抱歉，我现在无法处理这个请求")]}

        # 1. 从 state 和 item_info 收集模板变量
        item_info = state.get("item_info") or {}
        variables: dict[str, str] = {}

        # item_info 里的所有 key 都可作为模板变量
        for key, value in item_info.items():
            variables[key] = str(value) if value is not None else ""

        # state_hooks 优先覆盖（直接从 AgentState 读取）
        for hook in skill.state_hooks:
            value = state.get(hook)
            if value is None:
                value = item_info.get(hook, "")
            variables[hook] = str(value) if value is not None else ""

        # 2. 注入模板变量到 prompt
        skill_prompt = skill.prompt
        for key, value in variables.items():
            skill_prompt = skill_prompt.replace(f"{{{key}}}", value)

        # 未替换的占位符记录警告
        remaining = re.findall(r"\{(\w+)\}", skill_prompt)
        if remaining:
            logger.warning(f"skill '{intent}' prompt 中有未替换的变量: {remaining}")

        prompt = f"{global_rules}\n\n{skill_prompt}" if global_rules else skill_prompt

        # 3. 调用 LLM
        messages = [SystemMessage(content=prompt), *state["messages"]]

        try:
            response = await llm_client.invoke(messages, allow_empty=False)
            safe_content = check_safety(response.content)
            logger.info(f"skill_executor [{intent}] 执行成功，回复长度={len(safe_content)}")

            # 4. 构建返回 state
            updates: dict = {"messages": [AIMessage(content=safe_content)]}

            # 5. write_hooks 写回：bargain_count +1
            if "bargain_count" in skill.write_hooks:
                try:
                    updates["bargain_count"] = int(state.get("bargain_count", 0)) + 1
                except (TypeError, ValueError) as e:
                    logger.error(f"bargain_count 写回失败，保持原值: {e}")

            return updates

        except Exception as e:
            logger.error(f"skill_executor [{intent}] 执行失败: {e}")
            return {"messages": [AIMessage(content="抱歉，处理请求时出错了")]}

    return skill_executor_node
