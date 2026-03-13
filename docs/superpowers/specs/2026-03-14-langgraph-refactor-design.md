# LangGraph 重构设计文档

**日期：** 2026-03-14
**状态：** 已批准
**作者：** Claude + 用户协作设计

## 1. 概述

将当前基于简单继承的 Agent 系统重构为 LangGraph 状态图架构，实现图结构控制流、会话隔离和现代化技术栈。

### 1.1 核心目标

- **图结构** - 支持条件分支、循环等复杂流程
- **会话隔离** - 多用户并发场景下独立的状态管理
- **监控就绪** - 预留 LangSmith 集成接口
- **模块化** - 清晰的代码组织，便于扩展

### 1.2 当前架构问题

- 简单的三层路由（关键词 → LLM分类 → Agent分发）
- 无状态管理，会话隔离困难
- 缺少可视化和调试工具
- 扩展复杂流程需要大量自研工作

## 2. 系统架构

### 2.1 整体架构

```
当前架构：
用户消息 → 关键词匹配 → LLM分类 → Agent分发 → 生成回复

新架构（LangGraph StateGraph）：
用户消息 → 输入节点 → 意图识别节点 → 条件路由 →
    ├─ 议价节点 → [循环检查] → 条件边 → 输出节点
    ├─ 商品节点 → 输出节点
    └─ 默认节点 → 输出节点
```

### 2.2 文件结构

```
agents/
├── __init__.py
├── state.py              # 状态定义
├── graph.py              # 图构建
├── nodes/                # 节点目录
│   ├── __init__.py
│   ├── classify.py       # 意图识别节点
│   ├── price.py          # 议价节点
│   ├── product.py        # 商品咨询节点
│   └── default.py        # 默认节点
├── routers/              # 路由逻辑
│   ├── __init__.py
│   └── intent_router.py  # 意图路由条件边
└── utils.py              # 工具函数

services/
├── llm_client.py         # LLM 客户端
└── tools.py              # 外部工具调用
```

## 3. 核心组件设计

### 3.1 状态定义

```python
# agents/state.py

from typing import Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import TypedDict


class ItemInfo(BaseModel):
    """商品信息"""
    item_id: str = Field(description="商品ID")
    title: str = Field(description="商品标题")
    price: float = Field(description="商品价格")
    min_price: Optional[float] = Field(None, description="最低价")
    description: Optional[str] = Field(None, description="商品描述")


class AgentState(TypedDict):
    """Agent 状态定义"""
    messages: Annotated[Sequence[BaseMessage], "对话消息列表"]
    user_id: str
    intent: str
    bargain_count: int
    item_info: Optional[ItemInfo]
    manual_mode: bool
```

### 3.2 LLM 客户端封装

```python
# services/llm_client.py

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import os


class LLMError(Exception):
    """LLM 调用失败"""
    pass


class LLMClient:
    """LLM 客户端封装层 - 统一调用接口，支持重试和降级"""

    def __init__(self):
        # 构建 LLM 列表，按优先级排序
        self.llm_chain = self._build_llm_chain()

    def _build_llm_chain(self) -> list[ChatOpenAI]:
        """构建 LLM 调用链，按优先级排序"""
        chain = []

        # 主 LLM
        primary = self._create_llm(
            model=os.getenv("PRIMARY_MODEL"),
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("MODEL_BASE_URL"),
        )
        if primary:
            chain.append(primary)

        # 备用 LLM（可选）
        fallback = self._create_llm(
            model=os.getenv("FALLBACK_MODEL"),
            api_key=os.getenv("FALLBACK_API_KEY"),
            base_url=os.getenv("FALLBACK_BASE_URL"),
        )
        if fallback:
            chain.append(fallback)

        return chain

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def invoke(
        self,
        messages: list[BaseMessage],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> BaseMessage:
        """
        调用 LLM，支持自动重试和降级

        按优先级依次尝试 LLM，返回第一个成功的响应
        """
        # 遍历 LLM 链，返回第一个成功的
        for idx, llm in enumerate(self.llm_chain):
            try:
                response = await llm.ainvoke(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30,
                )
                logger.debug(f"LLM[{idx}] 调用成功")
                return response

            except Exception as e:
                logger.warning(f"LLM[{idx}] 调用失败: {e}")
                # 继续尝试下一个
                continue

        # 所有 LLM 都失败，返回兜底回复
        logger.error("所有 LLM 都失败，使用兜底回复")
        return self._get_fallback_response()

    def _create_llm(
        self,
        model: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str]
    ) -> Optional[ChatOpenAI]:
        """创建 ChatOpenAI 实例"""
        if not model or not api_key:
            return None

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.7,
            max_tokens=500,
        )

    def _get_fallback_response(self) -> BaseMessage:
        """获取兜底回复"""
        from langchain_core.messages import AIMessage

        fallback_msg = os.getenv(
            "FALLBACK_REPLY",
            "卖家暂时离开了，回来马上回复！"
        )
        return AIMessage(content=fallback_msg)
```

**关键特性：**

1. **重试机制** - 使用 `tenacity` 库，指数退避重试 3 次
2. **降级策略** - 主 LLM 失败自动切换到备用 LLM
3. **超时控制** - 30 秒超时防止长时间等待
4. **统一接口** - 所有节点通过 `LLMClient.invoke()` 调用
5. **监控埋点** - 预留接口用于 LangSmith 追踪
6. **兜底回复** - 最终失败返回友好的默认消息

### 3.3 节点实现

#### 3.3.1 意图识别节点

```python
# agents/nodes/classify.py

from langchain_core.messages import SystemMessage
from agents.state import AgentState
from services.llm_client import LLMClient


async def classify_node(state: AgentState) -> AgentState:
    """意图识别节点"""
    llm_client = LLMClient()

    with open("config/prompts/classify_prompt.md", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"]
    ]

    response = await llm_client.invoke(messages, temperature=0.3)

    intent = response.content.strip().lower()
    valid_intents = {"price", "product", "default", "no_reply"}
    intent = intent if intent in valid_intents else "default"

    return {"intent": intent}
```

#### 3.3.2 议价节点

```python
# agents/nodes/price.py

from langchain_core.messages import AIMessage, SystemMessage
from agents.state import AgentState
from services.llm_client import LLMClient
from services.tools import check_safety


async def price_node(state: AgentState) -> AgentState:
    """议价节点"""
    llm_client = LLMClient()

    with open("config/prompts/price_prompt.md", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    item_info = state["item_info"] or {}
    prompt = prompt_template.replace("{min_price}", str(item_info.get("min_price", "")))
    prompt = prompt.replace("{bargain_count}", str(state["bargain_count"]))

    messages = [
        SystemMessage(content=prompt),
        *state["messages"]
    ]

    temperature = min(0.3 + state["bargain_count"] * 0.15, 0.9)
    response = await llm_client.invoke(messages, temperature=temperature)

    safe_content = check_safety(response.content)

    return {
        "messages": [AIMessage(content=safe_content)],
        "bargain_count": state["bargain_count"] + 1
    }
```

#### 3.3.3 商品咨询节点

```python
# agents/nodes/product.py

from langchain_core.messages import AIMessage, SystemMessage
from agents.state import AgentState
from services.llm_client import LLMClient
from services.tools import check_safety


async def product_node(state: AgentState) -> AgentState:
    """商品咨询节点"""
    llm_client = LLMClient()

    with open("config/prompts/product_prompt.md", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"]
    ]

    response = await llm_client.invoke(messages, temperature=0.4)
    safe_content = check_safety(response.content)

    return {"messages": [AIMessage(content=safe_content)]}
```

#### 3.3.4 默认节点

```python
# agents/nodes/default.py

from langchain_core.messages import AIMessage, SystemMessage
from agents.state import AgentState
from services.llm_client import LLMClient
from services.tools import check_safety


async def default_node(state: AgentState) -> AgentState:
    """默认回复节点"""
    llm_client = LLMClient()

    with open("config/prompts/default_prompt.md", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"]
    ]

    response = await llm_client.invoke(messages, temperature=0.7)
    safe_content = check_safety(response.content)

    return {"messages": [AIMessage(content=safe_content)]}
```

### 3.4 路由逻辑

```python
# agents/routers/intent_router.py

from agents.state import AgentState


def route_intent(state: AgentState) -> str:
    """根据意图路由到不同节点"""

    intent = state.get("intent", "default")

    if intent == "no_reply":
        return "no_reply"

    return intent


def check_bargain_continue(state: AgentState) -> str:
    """检查是否继续议价"""

    if state["bargain_count"] >= 5:
        return "stop"

    last_message = state["messages"][-1]
    if hasattr(last_message, 'content'):
        content = last_message.content.lower()
        price_keywords = ["便宜", "价", "砍价", "少点", "多少钱", "最低"]
        if any(kw in content for kw in price_keywords):
            return "continue"

    return "stop"
```

### 3.5 图构建

```python
# agents/graph.py

from langgraph.graph import StateGraph, END
from langchain.checkpoint.sqlite import SqliteSaver
from agents.state import AgentState
from agents.nodes import classify_node, price_node, product_node, default_node
from agents.routers import route_intent, check_bargain_continue


def create_agent_graph():
    """创建 Agent 图"""

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("classify", classify_node)
    workflow.add_node("price", price_node)
    workflow.add_node("product", product_node)
    workflow.add_node("default", default_node)

    # 设置入口
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
        }
    )

    # 议价节点添加循环边
    workflow.add_conditional_edges(
        "price",
        check_bargain_continue,
        {
            "continue": "price",
            "stop": END,
        }
    )

    # 其他节点直接结束
    workflow.add_edge("product", END)
    workflow.add_edge("default", END)

    # 配置 checkpointer
    checkpointer = SqliteSaver.from_conn_string("data/chat_history.db")

    return workflow.compile(checkpointer=checkpointer)
```

### 3.6 工具函数

```python
# services/tools.py

def check_safety(text: str) -> str:
    """安全过滤"""
    blocked_phrases = ["微信", "QQ", "支付宝", "银行卡", "线下"]

    for phrase in blocked_phrases:
        if phrase in text:
            return "[安全提醒] 请通过闲鱼平台沟通，不要在站外交易哦"

    return text


async def get_item_info(item_id: str) -> dict:
    """查询商品信息（工具）"""
    import yaml

    with open("config/products.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for product in config.get("products", []):
        if product["item_id"] == item_id:
            return product

    return {}
```

### 3.7 错误处理与监控

```python
# agents/utils.py

from loguru import logger
from langchain_core.messages import AIMessage
import os


def handle_node_error(node_name: str, error: Exception) -> dict:
    """统一处理节点错误"""

    logger.error(f"节点 [{node_name}] 执行失败: {error}")

    # 返回兜底响应
    fallback_reply = os.getenv("FALLBACK_REPLY", "卖家暂时离开了，回来马上回复！")

    return {
        "messages": [AIMessage(content=fallback_reply)]
    }


def setup_langsmith_tracing():
    """配置 LangSmith 追踪（可选）"""

    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    if not langsmith_api_key:
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "goofish-customer")

    logger.info("LangSmith 追踪已启用")
```

## 4. 会话管理

### 4.1 会话隔离机制

- 使用 `thread_id` 作为会话标识符
- 每个用户会话独立存储在 SQLite
- 自动保存对话历史和状态

```python
# 使用示例
from agents.graph import create_agent_graph

graph = create_agent_graph()

# 用户 A 的会话
config_a = {"configurable": {"thread_id": "user_123"}}
response = await graph.ainvoke(input_data, config=config_a)

# 用户 B 的会话（完全独立）
config_b = {"configurable": {"thread_id": "user_456"}}
response = await graph.ainvoke(input_data, config=config_b)
```

### 4.2 状态持久化

- 自动保存到 SQLite（`data/chat_history.db`）
- 支持暂停和恢复对话
- 时间旅行调试（LangGraph 特性）

## 5. 环境配置

### 5.1 新增环境变量

```bash
# .env

# 主 LLM 配置
PRIMARY_MODEL=qwen-max
API_KEY=your_api_key
MODEL_BASE_URL=https://example.com/v1

# 备用 LLM 配置（可选）
FALLBACK_MODEL=gpt-3.5-turbo
FALLBACK_API_KEY=your_fallback_key
FALLBACK_BASE_URL=https://api.openai.com/v1

# 兜底回复
FALLBACK_REPLY=卖家暂时离开了，回来马上回复！

# LangSmith 监控（可选）
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=goofish-customer

# 功能开关
USE_LANGGRAPH=false  # 迁移期间使用
```

## 6. 迁移计划

### 6.1 阶段 1：并行运行

- 保留现有的 `IntentRouter` 和 Agents
- 新建 `agents/` 目录实现 LangGraph 版本
- 两者并行运行，对比结果

### 6.2 阶段 2：灰度切换

- 添加配置开关控制使用哪个版本
```python
USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "false").lower() == "true"

async def route_message(user_msg: str, user_id: str, ...):
    if USE_LANGGRAPH:
        return await langgraph_router.route(user_msg, user_id, ...)
    else:
        return await legacy_router.route(user_msg, ...)
```

### 6.3 阶段 3：完全迁移

- 确认 LangGraph 版本稳定后
- 移除旧代码
- 更新测试

## 7. 测试策略

### 7.1 单元测试

- 每个节点独立测试
- 路由逻辑测试
- LLMClient 降级测试

### 7.2 集成测试

- 完整图流程测试
- 会话隔离测试
- 循环节点测试

### 7.3 测试示例

```python
# tests/test_langgraph_nodes.py

import pytest
from agents.nodes.classify import classify_node
from agents.state import AgentState
from langchain_core.messages import HumanMessage


@pytest.mark.asyncio
async def test_classify_node():
    """测试意图识别节点"""
    state = AgentState(
        messages=[HumanMessage(content="能便宜点吗")],
        user_id="test_user",
        intent="",
        bargain_count=0,
        item_info=None,
        manual_mode=False,
    )

    result = await classify_node(state)
    assert result["intent"] == "price"


@pytest.mark.asyncio
async def test_graph_with_checkpointer():
    """测试会话隔离"""
    from agents.graph import create_agent_graph

    graph = create_agent_graph()

    # 用户 A 的会话
    config_a = {"configurable": {"thread_id": "user_a"}}
    state_a = await graph.ainvoke(
        {"messages": [HumanMessage(content="能便宜点吗")]},
        config=config_a
    )

    # 用户 B 的会话（独立状态）
    config_b = {"configurable": {"thread_id": "user_b"}}
    state_b = await graph.ainvoke(
        {"messages": [HumanMessage(content="什么型号")]},
        config=config_b
    )

    # 验证状态隔离
    assert state_a["bargain_count"] != state_b.get("bargain_count", 0)
```

## 8. 依赖更新

### 8.1 新增依赖

```txt
# requirements.txt 新增

langgraph>=0.2.0
langchain-openai>=0.2.0
langchain-core>=0.3.0
tenacity>=8.0.0
```

## 9. 后续扩展

### 9.1 短期扩展

- 添加更多工具（查询商品、发送消息等）
- 实现 Agent 协作
- 优化 prompt 模板

### 9.2 长期规划

- 接入 LangSmith 进行可视化监控
- 支持多 Agent 协作模式
- 添加长期记忆机制

## 10. 风险与挑战

### 10.1 技术风险

- LangGraph 学习曲线
- 迁移期间的稳定性
- 性能影响

### 10.2 缓解措施

- 分阶段迁移，保持向后兼容
- 充分的测试覆盖
- 保留旧系统作为备份

## 11. 总结

本设计通过引入 LangGraph 实现了：

1. ✅ **图结构控制流** - 声明式定义复杂业务逻辑
2. ✅ **会话隔离** - 多用户并发场景的独立状态管理
3. ✅ **现代化技术栈** - 为未来扩展奠定基础
4. ✅ **监控就绪** - LangSmith 集成点预留
5. ✅ **模块化设计** - 清晰的代码组织

该架构能够满足当前需求，同时为未来的功能扩展提供良好的基础。
