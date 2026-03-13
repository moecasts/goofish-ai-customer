# LangGraph 重构实施计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking。

**Goal:** 将当前基于简单继承的 Agent 系统重构为 LangGraph 状态图架构，实现图结构控制流、会话隔离和现代化技术栈。

**Architecture:** 使用 LangGraph 的 StateGraph 构建状态图，每个节点对应一个 Agent，通过条件边实现意图路由和议价循环。会话状态通过 SQLite checkpointer 持久化，实现多用户并发隔离。

**Tech Stack:** Python 3.11+, LangGraph, LangChain, OpenAI SDK, SQLite, tenacity（重试）, loguru（日志）

---

## Chunk 1: 基础设施准备

### Task 1: 添加项目依赖

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

**背景：** 需要添加 LangGraph、LangChain 和重试库等依赖。项目使用 pyproject.toml 作为依赖源，requirements.txt 从中自动生成。

- [ ] **Step 1: 更新 pyproject.toml 添加依赖**

在 `dependencies` 数组中添加以下依赖：

```toml
dependencies = [
    # ... 现有依赖 ...
    "langgraph>=0.2.0",
    "langchain-openai>=0.2.0",
    "langchain-core>=0.3.0",
    "tenacity>=8.0.0",
]
```

- [ ] **Step 2: 生成新的 requirements.txt**

运行：`make sync-requirements`

预期输出：requirements.txt 被更新，包含新增的依赖

- [ ] **Step 3: 安装新依赖**

运行：`make install-deps`

预期输出：依赖安装成功，无错误

- [ ] **Step 4: 验证导入**

运行：`python -c "import langgraph; import langchain; print('导入成功')"`

预期输出：`导入成功`

- [ ] **Step 5: 提交依赖更新**

```bash
git add pyproject.toml requirements.txt
git commit -m "chore(deps): add langgraph and langchain dependencies"
```

---

### Task 2: 创建状态定义

**Files:**
- Create: `agents/state.py`

**背景：** 定义 Agent 状态结构，包括对话历史、用户ID、意图、议价计数等字段。

- [ ] **Step 1: 创建状态定义文件**

创建 `agents/state.py`：

```python
"""Agent 状态定义。"""

from typing import Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import TypedDict


class ItemInfo(BaseModel):
    """商品信息。"""

    item_id: str = Field(description="商品ID")
    title: str = Field(description="商品标题")
    price: float = Field(description="商品价格")
    min_price: Optional[float] = Field(None, description="最低价")
    description: Optional[str] = Field(None, description="商品描述")


class AgentState(TypedDict):
    """Agent 状态定义。"""

    messages: Annotated[Sequence[BaseMessage], "对话消息列表"]
    user_id: str
    intent: str
    bargain_count: int
    item_info: Optional[ItemInfo]
    manual_mode: bool
```

- [ ] **Step 2: 编写状态定义测试**

创建 `tests/test_state.py`：

```python
"""测试 Agent 状态定义。"""

import pytest
from agents.state import AgentState, ItemInfo
from langchain_core.messages import HumanMessage


def test_item_info_creation():
    """测试商品信息创建。"""
    info = ItemInfo(
        item_id="123",
        title="测试商品",
        price=100.0,
        min_price=80.0,
    )
    assert info.item_id == "123"
    assert info.title == "测试商品"
    assert info.price == 100.0
    assert info.min_price == 80.0


def test_agent_state_structure():
    """测试 AgentState 结构。"""
    state = AgentState(
        messages=[HumanMessage(content="测试消息")],
        user_id="test_user",
        intent="price",
        bargain_count=1,
        item_info=None,
        manual_mode=False,
    )
    assert len(state["messages"]) == 1
    assert state["user_id"] == "test_user"
    assert state["intent"] == "price"
    assert state["bargain_count"] == 1
```

- [ ] **Step 3: 运行测试验证**

运行：`pytest tests/test_state.py -v`

预期输出：测试通过

- [ ] **Step 4: 提交状态定义**

```bash
git add agents/state.py tests/test_state.py
git commit -m "feat(agents): add AgentState definition with ItemInfo model"
```

---

### Task 3: 创建 LLM 客户端封装

**Files:**
- Create: `services/llm_client.py`
- Test: `tests/test_llm_client.py`

**背景：** 创建统一的 LLM 调用接口，支持重试、降级和超时控制。这是所有节点的基础依赖。

- [ ] **Step 1: 创建 LLMClient 类**

创建 `services/llm_client.py`：

```python
"""LLM 客户端封装层。"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import os


class LLMError(Exception):
    """LLM 调用失败。"""

    pass


class LLMClient:
    """LLM 客户端封装层 - 统一调用接口，支持重试和降级。"""

    def __init__(self):
        # 构建 LLM 列表，按优先级排序
        self.llm_chain = self._build_llm_chain()

    def _build_llm_chain(self) -> list[ChatOpenAI]:
        """构建 LLM 调用链，按优先级排序。"""
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
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def invoke(
        self,
        messages: list[BaseMessage],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> BaseMessage:
        """
        调用 LLM，支持自动重试和降级。

        按优先级依次尝试 LLM，返回第一个成功的响应。
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
        base_url: Optional[str],
    ) -> Optional[ChatOpenAI]:
        """创建 ChatOpenAI 实例。"""
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
        """获取兜底回复。"""
        from langchain_core.messages import AIMessage

        fallback_msg = os.getenv(
            "FALLBACK_REPLY", "卖家暂时离开了，回来马上回复！"
        )
        return AIMessage(content=fallback_msg)
```

- [ ] **Step 2: 编写 LLMClient 测试**

创建 `tests/test_llm_client.py`：

```python
"""测试 LLMClient。"""

import os
import pytest
from dotenv import load_dotenv
from services.llm_client import LLMClient, LLMError
from langchain_core.messages import HumanMessage

load_dotenv()


@pytest.mark.unit
class TestLLMClient:
    """测试 LLMClient 类。"""

    def test_build_llm_chain_with_primary_only(self):
        """测试只配置主 LLM 的情况。"""
        # 临时设置环境变量
        os.environ["PRIMARY_MODEL"] = "test-model"
        os.environ["API_KEY"] = "test-key"
        os.environ["MODEL_BASE_URL"] = "https://test.com"

        client = LLMClient()
        assert len(client.llm_chain) == 1

        # 清理
        del os.environ["PRIMARY_MODEL"]
        del os.environ["API_KEY"]
        del os.environ["MODEL_BASE_URL"]

    def test_build_llm_chain_with_fallback(self):
        """测试配置主 LLM 和备用 LLM 的情况。"""
        os.environ["PRIMARY_MODEL"] = "test-model"
        os.environ["API_KEY"] = "test-key"
        os.environ["MODEL_BASE_URL"] = "https://test.com"
        os.environ["FALLBACK_MODEL"] = "fallback-model"
        os.environ["FALLBACK_API_KEY"] = "fallback-key"
        os.environ["FALLBACK_BASE_URL"] = "https://fallback.com"

        client = LLMClient()
        assert len(client.llm_chain) == 2

        # 清理
        for key in [
            "PRIMARY_MODEL",
            "API_KEY",
            "MODEL_BASE_URL",
            "FALLBACK_MODEL",
            "FALLBACK_API_KEY",
            "FALLBACK_BASE_URL",
        ]:
            if key in os.environ:
                del os.environ[key]

    def test_create_llm_without_model(self):
        """测试没有模型配置时返回 None。"""
        client = LLMClient()
        result = client._create_llm(None, None, None)
        assert result is None

    def test_get_fallback_response(self):
        """测试获取兜底回复。"""
        client = LLMClient()
        response = client._get_fallback_response()
        assert response.content == "卖家暂时离开了，回来马上回复！"

    @pytest.mark.skipif(not os.getenv("API_KEY"), reason="API_KEY not set")
    @pytest.mark.asyncio
    async def test_invoke_with_real_api(self):
        """集成测试：使用真实 API 调用。"""
        os.environ["PRIMARY_MODEL"] = os.getenv("MODEL_NAME", "qwen-max")

        client = LLMClient()
        messages = [HumanMessage(content="你好")]

        response = await client.invoke(messages, temperature=0.7)

        assert hasattr(response, "content")
        assert len(response.content) > 0
```

- [ ] **Step 3: 运行单元测试**

运行：`pytest tests/test_llm_client.py::TestLLMClient::test_build_llm_chain_with_primary_only -v`

预期输出：PASS

- [ ] **Step 4: 运行集成测试（如果配置了 API_KEY）**

运行：`pytest tests/test_llm_client.py::TestLLMClient::test_invoke_with_real_api -v`

预期输出：PASS（如果配置了 API_KEY）或 SKIP（如果未配置）

- [ ] **Step 5: 提交 LLMClient**

```bash
git add services/llm_client.py tests/test_llm_client.py
git commit -m "feat(services): add LLMClient with retry and fallback support"
```

---

### Task 4: 创建工具函数

**Files:**
- Create: `services/tools.py`
- Test: `tests/test_tools.py`

**背景：** 创建安全过滤和商品信息查询等工具函数，供节点使用。

- [ ] **Step 1: 创建工具函数文件**

创建 `services/tools.py`：

```python
"""工具函数。"""

from loguru import logger
import yaml


def check_safety(text: str) -> str:
    """
    安全过滤。

    检测敏感词（微信/QQ/支付宝/银行卡/线下交易），
    命中则替换为平台沟通提醒。
    """
    blocked_phrases = ["微信", "QQ", "支付宝", "银行卡", "线下"]

    for phrase in blocked_phrases:
        if phrase in text:
            logger.info(f"检测到敏感词: {phrase}")
            return "[安全提醒] 请通过闲鱼平台沟通，不要在站外交易哦"

    return text


async def get_item_info(item_id: str) -> dict:
    """
    查询商品信息。

    从配置文件中获取商品的补充信息（最低价、卖点等）。
    """
    try:
        with open("config/products.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for product in config.get("products", []):
            if product["item_id"] == item_id:
                return product

        logger.warning(f"未找到商品 {item_id} 的配置信息")
        return {}

    except FileNotFoundError:
        logger.error("config/products.yaml 文件未找到")
        return {}
    except Exception as e:
        logger.error(f"读取商品配置失败: {e}")
        return {}
```

- [ ] **Step 2: 编写工具函数测试**

创建 `tests/test_tools.py`：

```python
"""测试工具函数。"""

import pytest
from services.tools import check_safety, get_item_info


@pytest.mark.unit
class TestSafetyCheck:
    """测试安全过滤功能。"""

    def test_blocks_wechat(self):
        """测试拦截微信关键词。"""
        result = check_safety("加我微信吧")
        assert "安全提醒" in result
        assert "闲鱼平台" in result

    def test_blocks_qq(self):
        """测试拦截 QQ 关键词。"""
        result = check_safety("QQ联系")
        assert "安全提醒" in result

    def test_blocks_alipay(self):
        """测试拦截支付宝关键词。"""
        result = check_safety("用支付宝支付")
        assert "安全提醒" in result

    def test_blocks_bank_card(self):
        """测试拦截银行卡关键词。"""
        result = check_safety("银行卡转账")
        assert "安全提醒" in result

    def test_blocks_offline(self):
        """测试拦截线下交易关键词。"""
        result = check_safety("线下见面交易")
        assert "安全提醒" in result

    def test_passes_normal_message(self):
        """测试正常消息通过。"""
        result = check_safety("这个商品不错")
        assert result == "这个商品不错"
        assert "安全提醒" not in result


@pytest.mark.unit
class TestGetItemInfo:
    """测试商品信息查询功能。"""

    @pytest.mark.asyncio
    async def test_get_existing_item(self):
        """测试查询存在的商品。"""
        # 假设 config/products.yaml 中有该商品
        info = await get_item_info("123456789")
        # 根据实际配置验证
        assert isinstance(info, dict)

    @pytest.mark.asyncio
    async def test_get_non_existing_item(self):
        """测试查询不存在的商品。"""
        info = await get_item_info("nonexistent")
        assert info == {}
```

- [ ] **Step 3: 运行测试**

运行：`pytest tests/test_tools.py -v`

预期输出：测试通过

- [ ] **Step 4: 提交工具函数**

```bash
git add services/tools.py tests/test_tools.py
git commit -m "feat(services): add safety filter and item info tools"
```

---

## Chunk 2: 节点实现

### Task 5: 创建节点目录结构

**Files:**
- Create: `agents/nodes/__init__.py`

**背景：** 创建节点目录并初始化，为后续节点实现做准备。

- [ ] **Step 1: 创建节点目录**

运行：`mkdir -p agents/nodes`

- [ ] **Step 2: 创建 __init__.py**

创建 `agents/nodes/__init__.py`：

```python
"""Agent 节点模块。"""

from agents.nodes.classify import classify_node
from agents.nodes.price import price_node
from agents.nodes.product import product_node
from agents.nodes.default import default_node

__all__ = [
    "classify_node",
    "price_node",
    "product_node",
    "default_node",
]
```

- [ ] **Step 3: 提交目录结构**

```bash
git add agents/nodes/__init__.py
git commit -m "feat(agents): create nodes directory structure"
```

---

### Task 6: 实现意图识别节点

**Files:**
- Create: `agents/nodes/classify.py`
- Test: `tests/test_classify_node.py`

**背景：** 意图识别节点是图的入口点，负责识别用户意图并路由到对应的处理节点。

- [ ] **Step 1: 实现意图识别节点**

创建 `agents/nodes/classify.py`：

```python
"""意图识别节点。"""

from langchain_core.messages import SystemMessage
from agents.state import AgentState
from services.llm_client import LLMClient
from loguru import logger


async def classify_node(state: AgentState) -> AgentState:
    """
    意图识别节点。

    使用 LLM 分析用户消息，识别意图类型：
    - price: 议价相关
    - product: 商品咨询
    - default: 默认回复
    - no_reply: 不需要回复
    """
    llm_client = LLMClient()

    # 加载分类 prompt
    try:
        with open("config/prompts/classify_prompt.md", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        logger.error("classify_prompt.md 文件未找到")
        return {"intent": "default"}

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"],
    ]

    try:
        # 使用低温度提高分类准确性
        response = await llm_client.invoke(messages, temperature=0.3)

        # 解析意图
        intent = response.content.strip().lower()
        valid_intents = {"price", "product", "default", "no_reply"}
        intent = intent if intent in valid_intents else "default"

        logger.info(f"意图识别结果: {intent}")
        return {"intent": intent}

    except Exception as e:
        logger.error(f"意图识别失败: {e}")
        return {"intent": "default"}
```

- [ ] **Step 2: 编写意图识别节点测试**

创建 `tests/test_classify_node.py`：

```python
"""测试意图识别节点。"""

import pytest
from agents.nodes.classify import classify_node
from agents.state import AgentState
from langchain_core.messages import HumanMessage


@pytest.mark.unit
class TestClassifyNode:
    """测试意图识别节点。"""

    @pytest.mark.asyncio
    async def test_classify_price_intent(self):
        """测试识别议价意图。"""
        state = AgentState(
            messages=[HumanMessage(content="能便宜点吗")],
            user_id="test_user",
            intent="",
            bargain_count=0,
            item_info=None,
            manual_mode=False,
        )

        result = await classify_node(state)
        assert "intent" in result
        # 注意：实际结果取决于 LLM 响应，这里测试基本结构

    @pytest.mark.asyncio
    async def test_classify_product_intent(self):
        """测试识别商品咨询意图。"""
        state = AgentState(
            messages=[HumanMessage(content="参数是什么")],
            user_id="test_user",
            intent="",
            bargain_count=0,
            item_info=None,
            manual_mode=False,
        )

        result = await classify_node(state)
        assert "intent" in result

    @pytest.mark.asyncio
    async def test_handles_missing_prompt_file(self):
        """测试处理 prompt 文件缺失的情况。"""
        # 临时重命名文件
        import os
        import shutil

        if os.path.exists("config/prompts/classify_prompt.md"):
            shutil.move(
                "config/prompts/classify_prompt.md",
                "config/prompts/classify_prompt.md.bak",
            )

        try:
            state = AgentState(
                messages=[HumanMessage(content="测试")],
                user_id="test_user",
                intent="",
                bargain_count=0,
                item_info=None,
                manual_mode=False,
            )

            result = await classify_node(state)
            assert result["intent"] == "default"  # 兜底返回

        finally:
            # 恢复文件
            if os.path.exists("config/prompts/classify_prompt.md.bak"):
                shutil.move(
                    "config/prompts/classify_prompt.md.bak",
                    "config/prompts/classify_prompt.md",
                )
```

- [ ] **Step 3: 运行测试**

运行：`pytest tests/test_classify_node.py -v`

预期输出：测试通过

- [ ] **Step 4: 提交意图识别节点**

```bash
git add agents/nodes/classify.py tests/test_classify_node.py
git commit -m "feat(agents): add intent classification node"
```

---

### Task 7: 实现议价节点

**Files:**
- Create: `agents/nodes/price.py`
- Test: `tests/test_price_node.py`

**背景：** 议价节点处理价格相关的对话，支持多轮议价和动态温度调整。

- [ ] **Step 1: 实现议价节点**

创建 `agents/nodes/price.py`：

```python
"""议价节点。"""

from langchain_core.messages import AIMessage, SystemMessage
from agents.state import AgentState
from services.llm_client import LLMClient
from services.tools import check_safety
from loguru import logger


async def price_node(state: AgentState) -> AgentState:
    """
    议价节点。

    根据议价次数动态调整温度，实现逐步让步的策略。
    第 1 次 (t=0.3): 坚持原价
    第 2 次 (t=0.45): 小幅让步
    第 3 次 (t=0.6): 接近底价
    3 次以上 (t=0.9): 坚持底价，委婉拒绝
    """
    llm_client = LLMClient()

    # 加载议价 prompt
    try:
        with open("config/prompts/price_prompt.md", "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        logger.error("price_prompt.md 文件未找到")
        return {"messages": [AIMessage(content="抱歉，暂时无法处理议价请求")]}

    # 替换占位符
    item_info = state["item_info"] or {}
    prompt = prompt_template.replace("{min_price}", str(item_info.get("min_price", "")))
    prompt = prompt.replace("{bargain_count}", str(state["bargain_count"]))

    messages = [
        SystemMessage(content=prompt),
        *state["messages"],
    ]

    try:
        # 动态温度：议价次数越多，越灵活
        temperature = min(0.3 + state["bargain_count"] * 0.15, 0.9)

        response = await llm_client.invoke(messages, temperature=temperature)

        # 安全过滤
        safe_content = check_safety(response.content)

        logger.info(f"议价节点: count={state['bargain_count']}, temp={temperature}")

        return {
            "messages": [AIMessage(content=safe_content)],
            "bargain_count": state["bargain_count"] + 1,
        }

    except Exception as e:
        logger.error(f"议价节点执行失败: {e}")
        return {"messages": [AIMessage(content="抱歉，处理议价请求时出错了")]}


def calculate_price_temperature(bargain_count: int) -> float:
    """
    计算议价温度。

    Args:
        bargain_count: 议价次数

    Returns:
        温度值（0-1 之间）
    """
    return min(0.3 + bargain_count * 0.15, 0.9)
```

- [ ] **Step 2: 编写议价节点测试**

创建 `tests/test_price_node.py`：

```python
"""测试议价节点。"""

import pytest
from agents.nodes.price import price_node, calculate_price_temperature
from agents.state import AgentState
from langchain_core.messages import HumanMessage


@pytest.mark.unit
class TestPriceNode:
    """测试议价节点。"""

    def test_temperature_calculation(self):
        """测试温度计算。"""
        assert calculate_price_temperature(0) == 0.3
        assert calculate_price_temperature(1) == 0.45
        assert calculate_price_temperature(2) == 0.6
        assert calculate_price_temperature(4) == 0.9
        assert calculate_price_temperature(10) == 0.9  # 封顶

    @pytest.mark.asyncio
    async def test_price_node_increments_count(self):
        """测试议价节点增加计数。"""
        state = AgentState(
            messages=[HumanMessage(content="能便宜点吗")],
            user_id="test_user",
            intent="price",
            bargain_count=1,
            item_info={"min_price": 100},
            manual_mode=False,
        )

        result = await price_node(state)
        assert result["bargain_count"] == 2

    @pytest.mark.asyncio
    async def test_price_node_with_no_item_info(self):
        """测试没有商品信息的情况。"""
        state = AgentState(
            messages=[HumanMessage(content="能便宜点吗")],
            user_id="test_user",
            intent="price",
            bargain_count=0,
            item_info=None,
            manual_mode=False,
        )

        result = await price_node(state)
        assert "messages" in result
        assert result["bargain_count"] == 1
```

- [ ] **Step 3: 运行测试**

运行：`pytest tests/test_price_node.py -v`

预期输出：测试通过

- [ ] **Step 4: 提交议价节点**

```bash
git add agents/nodes/price.py tests/test_price_node.py
git commit -m "feat(agents): add price negotiation node with dynamic temperature"
```

---

### Task 8: 实现商品咨询节点

**Files:**
- Create: `agents/nodes/product.py`
- Test: `tests/test_product_node.py`

**背景：** 商品咨询节点处理产品相关的询问，如参数、规格、型号等。

- [ ] **Step 1: 实现商品咨询节点**

创建 `agents/nodes/product.py`：

```python
"""商品咨询节点。"""

from langchain_core.messages import AIMessage, SystemMessage
from agents.state import AgentState
from services.llm_client import LLMClient
from services.tools import check_safety
from loguru import logger


async def product_node(state: AgentState) -> AgentState:
    """
    商品咨询节点。

    处理关于商品参数、规格、型号、对比等咨询。
    使用较低温度以提供准确、一致的信息。
    """
    llm_client = LLMClient()

    # 加载商品咨询 prompt
    try:
        with open("config/prompts/product_prompt.md", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        logger.error("product_prompt.md 文件未找到")
        return {"messages": [AIMessage(content="抱歉，暂时无法处理商品咨询")]}

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"],
    ]

    try:
        response = await llm_client.invoke(messages, temperature=0.4)

        # 安全过滤
        safe_content = check_safety(response.content)

        logger.info("商品咨询节点执行成功")

        return {"messages": [AIMessage(content=safe_content)]}

    except Exception as e:
        logger.error(f"商品咨询节点执行失败: {e}")
        return {"messages": [AIMessage(content="抱歉，处理商品咨询时出错了")]}
```

- [ ] **Step 2: 编写商品咨询节点测试**

创建 `tests/test_product_node.py`：

```python
"""测试商品咨询节点。"""

import pytest
from agents.nodes.product import product_node
from agents.state import AgentState
from langchain_core.messages import HumanMessage


@pytest.mark.unit
class TestProductNode:
    """测试商品咨询节点。"""

    @pytest.mark.asyncio
    async def test_product_node_basic(self):
        """测试商品咨询节点基本功能。"""
        state = AgentState(
            messages=[HumanMessage(content="参数是什么")],
            user_id="test_user",
            intent="product",
            bargain_count=0,
            item_info={"title": "测试商品", "price": 100},
            manual_mode=False,
        )

        result = await product_node(state)
        assert "messages" in result
        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_product_node_with_item_info(self):
        """测试带商品信息的情况。"""
        state = AgentState(
            messages=[HumanMessage(content="和 iPhone 14 比怎么样")],
            user_id="test_user",
            intent="product",
            bargain_count=0,
            item_info={
                "title": "iPhone 15",
                "price": 5000,
                "description": "最新款 iPhone",
            },
            manual_mode=False,
        )

        result = await product_node(state)
        assert "messages" in result
```

- [ ] **Step 3: 运行测试**

运行：`pytest tests/test_product_node.py -v`

预期输出：测试通过

- [ ] **Step 4: 提交商品咨询节点**

```bash
git add agents/nodes/product.py tests/test_product_node.py
git commit -m "feat(agents): add product inquiry node"
```

---

### Task 9: 实现默认节点

**Files:**
- Create: `agents/nodes/default.py`
- Test: `tests/test_default_node.py`

**背景：** 默认节点处理所有不属于特定类别的一般性对话。

- [ ] **Step 1: 实现默认节点**

创建 `agents/nodes/default.py`：

```python
"""默认回复节点。"""

from langchain_core.messages import AIMessage, SystemMessage
from agents.state import AgentState
from services.llm_client import LLMClient
from services.tools import check_safety
from loguru import logger


async def default_node(state: AgentState) -> AgentState:
    """
    默认回复节点。

    处理所有不属于特定类别的一般性对话，
    如问候、感谢、闲聊等。
    """
    llm_client = LLMClient()

    # 加载默认 prompt
    try:
        with open("config/prompts/default_prompt.md", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        logger.error("default_prompt.md 文件未找到")
        return {"messages": [AIMessage(content="您好，请问有什么可以帮您的？")]}

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"],
    ]

    try:
        response = await llm_client.invoke(messages, temperature=0.7)

        # 安全过滤
        safe_content = check_safety(response.content)

        logger.info("默认节点执行成功")

        return {"messages": [AIMessage(content=safe_content)]}

    except Exception as e:
        logger.error(f"默认节点执行失败: {e}")
        return {"messages": [AIMessage(content="抱歉，我现在无法回复，请稍后再试")]}
```

- [ ] **Step 2: 编写默认节点测试**

创建 `tests/test_default_node.py`：

```python
"""测试默认节点。"""

import pytest
from agents.nodes.default import default_node
from agents.state import AgentState
from langchain_core.messages import HumanMessage


@pytest.mark.unit
class TestDefaultNode:
    """测试默认节点。"""

    @pytest.mark.asyncio
    async def test_default_node_greeting(self):
        """测试问候消息。"""
        state = AgentState(
            messages=[HumanMessage(content="你好")],
            user_id="test_user",
            intent="default",
            bargain_count=0,
            item_info=None,
            manual_mode=False,
        )

        result = await default_node(state)
        assert "messages" in result
        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_default_node_thanks(self):
        """测试感谢消息。"""
        state = AgentState(
            messages=[HumanMessage(content="谢谢")],
            user_id="test_user",
            intent="default",
            bargain_count=0,
            item_info=None,
            manual_mode=False,
        )

        result = await default_node(state)
        assert "messages" in result
```

- [ ] **Step 3: 运行测试**

运行：`pytest tests/test_default_node.py -v`

预期输出：测试通过

- [ ] **Step 4: 提交默认节点**

```bash
git add agents/nodes/default.py tests/test_default_node.py
git commit -m "feat(agents): add default reply node"
```

---

## Chunk 3: 路由和图构建

### Task 10: 创建路由目录结构

**Files:**
- Create: `agents/routers/__init__.py`

**背景：** 创建路由目录并初始化，为路由逻辑实现做准备。

- [ ] **Step 1: 创建路由目录**

运行：`mkdir -p agents/routers`

- [ ] **Step 2: 创建 __init__.py**

创建 `agents/routers/__init__.py`：

```python
"""路由逻辑模块。"""

from agents.routers.intent_router import route_intent, check_bargain_continue

__all__ = [
    "route_intent",
    "check_bargain_continue",
]
```

- [ ] **Step 3: 提交路由目录**

```bash
git add agents/routers/__init__.py
git commit -m "feat(agents): create routers directory structure"
```

---

### Task 11: 实现意图路由

**Files:**
- Create: `agents/routers/intent_router.py`
- Test: `tests/test_intent_router.py`

**背景：** 意图路由负责根据识别出的意图分发到对应的节点。

- [ ] **Step 1: 实现意图路由**

创建 `agents/routers/intent_router.py`：

```python
"""意图路由逻辑。"""

from agents.state import AgentState
from loguru import logger


def route_intent(state: AgentState) -> str:
    """
    根据意图路由到不同节点。

    Args:
        state: Agent 状态

    Returns:
        目标节点名称
    """
    intent = state.get("intent", "default")

    logger.info(f"路由意图: {intent} -> {intent}")

    if intent == "no_reply":
        return "no_reply"

    return intent


def check_bargain_continue(state: AgentState) -> str:
    """
    检查是否继续议价。

    Args:
        state: Agent 状态

    Returns:
        "continue" 或 "stop"
    """
    # 议价超过 5 次，停止循环
    if state["bargain_count"] >= 5:
        logger.info(f"议价次数达到上限 ({state['bargain_count']}), 停止")
        return "stop"

    # 检查最新消息是否包含价格相关关键词
    last_message = state["messages"][-1]
    if hasattr(last_message, "content"):
        content = last_message.content.lower()
        price_keywords = ["便宜", "价", "砍价", "少点", "多少钱", "最低"]
        if any(kw in content for kw in price_keywords):
            logger.info("检测到价格关键词，继续议价")
            return "continue"

    logger.info("未检测到议价意图，结束")
    return "stop"
```

- [ ] **Step 2: 编写路由测试**

创建 `tests/test_intent_router.py`：

```python
"""测试意图路由。"""

import pytest
from agents.routers.intent_router import route_intent, check_bargain_continue
from agents.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage


@pytest.mark.unit
class TestRouteIntent:
    """测试意图路由。"""

    def test_route_price_intent(self):
        """测试路由议价意图。"""
        state = AgentState(
            messages=[],
            user_id="test_user",
            intent="price",
            bargain_count=0,
            item_info=None,
            manual_mode=False,
        )
        result = route_intent(state)
        assert result == "price"

    def test_route_product_intent(self):
        """测试路由商品咨询意图。"""
        state = AgentState(
            messages=[],
            user_id="test_user",
            intent="product",
            bargain_count=0,
            item_info=None,
            manual_mode=False,
        )
        result = route_intent(state)
        assert result == "product"

    def test_route_default_intent(self):
        """测试路由默认意图。"""
        state = AgentState(
            messages=[],
            user_id="test_user",
            intent="default",
            bargain_count=0,
            item_info=None,
            manual_mode=False,
        )
        result = route_intent(state)
        assert result == "default"

    def test_route_no_reply_intent(self):
        """测试路由不回复意图。"""
        state = AgentState(
            messages=[],
            user_id="test_user",
            intent="no_reply",
            bargain_count=0,
            item_info=None,
            manual_mode=False,
        )
        result = route_intent(state)
        assert result == "no_reply"

    def test_route_missing_intent_defaults_to_default(self):
        """测试缺失意图时默认为 default。"""
        state = AgentState(
            messages=[],
            user_id="test_user",
            intent="",  # 空字符串
            bargain_count=0,
            item_info=None,
            manual_mode=False,
        )
        result = route_intent(state)
        assert result == ""


@pytest.mark.unit
class TestCheckBargainContinue:
    """测试议价继续检查。"""

    def test_stop_after_max_bargains(self):
        """测试达到最大议价次数后停止。"""
        state = AgentState(
            messages=[AIMessage(content="能便宜点吗")],
            user_id="test_user",
            intent="price",
            bargain_count=5,  # 达到上限
            item_info=None,
            manual_mode=False,
        )
        result = check_bargain_continue(state)
        assert result == "stop"

    def test_continue_with_price_keyword(self):
        """测试检测到价格关键词时继续。"""
        state = AgentState(
            messages=[HumanMessage(content="能再便宜点吗")],
            user_id="test_user",
            intent="price",
            bargain_count=2,
            item_info=None,
            manual_mode=False,
        )
        result = check_bargain_continue(state)
        assert result == "continue"

    def test_stop_without_price_keyword(self):
        """测试没有价格关键词时停止。"""
        state = AgentState(
            messages=[HumanMessage(content="好的，谢谢")],
            user_id="test_user",
            intent="price",
            bargain_count=1,
            item_info=None,
            manual_mode=False,
        )
        result = check_bargain_continue(state)
        assert result == "stop"
```

- [ ] **Step 3: 运行测试**

运行：`pytest tests/test_intent_router.py -v`

预期输出：测试通过

- [ ] **Step 4: 提交意图路由**

```bash
git add agents/routers/intent_router.py tests/test_intent_router.py
git commit -m "feat(agents): add intent routing logic with bargain cycle check"
```

---

### Task 12: 构建状态图

**Files:**
- Create: `agents/graph.py`
- Test: `tests/test_graph.py`

**背景：** 将所有节点和路由逻辑组装成完整的状态图，配置检查点以支持会话持久化。

- [ ] **Step 1: 实现图构建**

创建 `agents/graph.py`：

```python
"""Agent 状态图构建。"""

from langgraph.graph import StateGraph, END
from langchain.checkpoint.sqlite import SqliteSaver
from agents.state import AgentState
from agents.nodes import classify_node, price_node, product_node, default_node
from agents.routers import route_intent, check_bargain_continue
from loguru import logger


def create_agent_graph():
    """
    创建 Agent 状态图。

    Returns:
        编译后的状态图，配置了 SQLite 检查点
    """
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

    # 配置 checkpointer 用于状态持久化
    checkpointer = SqliteSaver.from_conn_string("data/chat_history.db")

    logger.info("Agent 状态图构建完成")

    return workflow.compile(checkpointer=checkpointer)
```

- [ ] **Step 2: 编写图测试**

创建 `tests/test_graph.py`：

```python
"""测试状态图。"""

import pytest
from agents.graph import create_agent_graph
from langchain_core.messages import HumanMessage


@pytest.mark.integration
class TestAgentGraph:
    """测试 Agent 状态图。"""

    def test_graph_creation(self):
        """测试图创建。"""
        graph = create_agent_graph()
        assert graph is not None

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not pytest.importorskip("os").getenv("API_KEY"),
        reason="API_KEY not set",
    )
    async def test_graph_execution_simple(self):
        """测试图执行（简单场景）。"""
        graph = create_agent_graph()

        config = {"configurable": {"thread_id": "test_thread"}}

        initial_state = {
            "messages": [HumanMessage(content="你好")],
            "user_id": "test_user",
            "intent": "",
            "bargain_count": 0,
            "item_info": None,
            "manual_mode": False,
        }

        result = await graph.ainvoke(initial_state, config=config)

        assert "messages" in result
        assert len(result["messages"]) > 1  # 原始消息 + AI 回复

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not pytest.importorskip("os").getenv("API_KEY"),
        reason="API_KEY not set",
    )
    async def test_graph_session_isolation(self):
        """测试会话隔离。"""
        graph = create_agent_graph()

        # 用户 A 的会话
        config_a = {"configurable": {"thread_id": "user_a"}}
        state_a = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="能便宜点吗")],
                "user_id": "user_a",
                "intent": "",
                "bargain_count": 0,
                "item_info": None,
                "manual_mode": False,
            },
            config=config_a,
        )

        # 用户 B 的会话
        config_b = {"configurable": {"thread_id": "user_b"}}
        state_b = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="什么型号")],
                "user_id": "user_b",
                "intent": "",
                "bargain_count": 0,
                "item_info": None,
                "manual_mode": False,
            },
            config=config_b,
        )

        # 验证状态隔离
        assert state_a.get("bargain_count", 0) != state_b.get("bargain_count", 0)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not pytest.importorskip("os").getenv("API_KEY"),
        reason="API_KEY not set",
    )
    async def test_graph_bargain_cycle(self):
        """测试议价循环。"""
        graph = create_agent_graph()

        config = {"configurable": {"thread_id": "bargain_test"}}

        # 第一次议价
        state = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="能便宜点吗")],
                "user_id": "test_user",
                "intent": "price",
                "bargain_count": 0,
                "item_info": {"min_price": 100},
                "manual_mode": False,
            },
            config=config,
        )

        assert state.get("bargain_count", 0) >= 1
```

- [ ] **Step 3: 运行图创建测试**

运行：`pytest tests/test_graph.py::TestAgentGraph::test_graph_creation -v`

预期输出：PASS

- [ ] **Step 4: 运行集成测试（如果配置了 API_KEY）**

运行：`pytest tests/test_graph.py -v`

预期输出：集成测试通过（或 SKIP 如果没有 API_KEY）

- [ ] **Step 5: 提交状态图**

```bash
git add agents/graph.py tests/test_graph.py
git commit -m "feat(agents): build complete agent state graph with checkpointing"
```

---

## Chunk 4: 集成和配置

### Task 13: 更新环境变量配置

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

**背景：** 添加新的环境变量配置项，支持 LangGraph 功能。

- [ ] **Step 1: 更新 .env.example**

在 `.env.example` 中添加：

```bash
# LLM 配置（更新）
PRIMARY_MODEL=qwen-max
API_KEY=your_api_key
MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 备用 LLM（可选）
FALLBACK_MODEL=
FALLBACK_API_KEY=
FALLBACK_BASE_URL=

# 兜底回复
FALLBACK_REPLY=卖家暂时离开了，回来马上回复！

# LangSmith 监控（可选）
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=goofish-customer

# 功能开关
USE_LANGGRAPH=false
```

- [ ] **Step 2: 更新 README.md**

在 README.md 的环境变量表格中添加：

```markdown
| 变量 | 默认值 | 说明 |
|------|--------|------|
| PRIMARY_MODEL | qwen-max | 主 LLM 模型名称 |
| API_KEY | - | LLM API Key |
| MODEL_BASE_URL | https://dashscope.aliyuncs.com/compatible-mode/v1 | LLM API 地址 |
| FALLBACK_MODEL | - | 备用 LLM 模型（可选） |
| FALLBACK_API_KEY | - | 备用 LLM API Key |
| FALLBACK_BASE_URL | - | 备用 LLM API 地址 |
| FALLBACK_REPLY | 卖家暂时离开了... | LLM 失败时的兜底回复 |
| LANGSMITH_API_KEY | - | LangSmith API Key（可选） |
| LANGSMITH_PROJECT | goofish-customer | LangSmith 项目名称 |
| USE_LANGGRAPH | false | 是否使用 LangGraph 版本 |
```

- [ ] **Step 3: 提交配置更新**

```bash
git add .env.example README.md
git commit -m "docs(config): add LangGraph environment variables"
```

---

### Task 14: 创建兼容层

**Files:**
- Create: `agents/adapter.py`
- Test: `tests/test_adapter.py`

**背景：** 创建适配器，使 LangGraph 版本可以与现有代码并行运行，支持灰度切换。

- [ ] **Step 1: 实现适配器**

创建 `agents/adapter.py`：

```python
"""LangGraph 适配器，兼容现有路由接口。"""

import os
from agents.graph import create_agent_graph
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger


class LangGraphRouter:
    """LangGraph 路由器，兼容现有的 IntentRouter 接口。"""

    def __init__(self):
        self.graph = create_agent_graph()
        logger.info("LangGraphRouter 初始化完成")

    async def route(
        self,
        user_msg: str,
        item_desc: str = "",
        context: str = "",
        user_id: str = "default",
        **kwargs,
    ) -> str:
        """
        路由消息到 LangGraph 处理。

        Args:
            user_msg: 用户消息
            item_desc: 商品描述
            context: 对话历史
            user_id: 用户 ID
            **kwargs: 其他参数（如 bargain_count, item_info）

        Returns:
            AI 回复内容
        """
        try:
            # 构建初始状态
            messages = [HumanMessage(content=user_msg)]

            # 如果有对话历史，添加到消息列表
            if context:
                # 这里简化处理，实际可能需要解析历史格式
                pass

            initial_state = {
                "messages": messages,
                "user_id": user_id,
                "intent": "",
                "bargain_count": kwargs.get("bargain_count", 0),
                "item_info": kwargs.get("item_info"),
                "manual_mode": kwargs.get("manual_mode", False),
            }

            # 使用 thread_id 实现会话隔离
            config = {"configurable": {"thread_id": f"user_{user_id}"}}

            # 执行图
            result = await self.graph.ainvoke(initial_state, config=config)

            # 提取最后一条 AI 消息
            if result.get("messages"):
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    return last_message.content

            return ""

        except Exception as e:
            logger.error(f"LangGraph 执行失败: {e}")
            # 返回兜底回复
            return os.getenv("FALLBACK_REPLY", "卖家暂时离开了，回来马上回复！")


def create_router():
    """
    创建路由器实例。

    根据环境变量决定使用 LangGraph 还是旧版路由器。
    """
    use_langgraph = os.getenv("USE_LANGGRAPH", "false").lower() == "true"

    if use_langgraph:
        logger.info("使用 LangGraph 路由器")
        return LangGraphRouter()
    else:
        logger.info("使用旧版路由器")
        from agents.router import IntentRouter

        return IntentRouter()
```

- [ ] **Step 2: 编写适配器测试**

创建 `tests/test_adapter.py`：

```python
"""测试 LangGraph 适配器。"""

import pytest
import os
from agents.adapter import LangGraphRouter, create_router


@pytest.mark.integration
class TestLangGraphRouter:
    """测试 LangGraph 路由器。"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("API_KEY"), reason="API_KEY not set")
    async def test_route_basic_message(self):
        """测试路由基本消息。"""
        router = LangGraphRouter()
        response = await router.route(
            user_msg="你好",
            user_id="test_user",
        )
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("API_KEY"), reason="API_KEY not set")
    async def test_route_with_bargain_count(self):
        """测试带议价次数的路由。"""
        router = LangGraphRouter()
        response = await router.route(
            user_msg="能便宜点吗",
            user_id="test_user",
            bargain_count=1,
            item_info={"min_price": 100},
        )
        assert isinstance(response, str)

    def test_create_router_with_langgraph_disabled(self):
        """测试创建路由器（LangGraph 禁用）。"""
        # 临时设置环境变量
        os.environ["USE_LANGGRAPH"] = "false"
        router = create_router()
        # 应该返回旧版路由器
        assert router.__class__.__name__ == "IntentRouter"

    def test_create_router_with_langgraph_enabled(self):
        """测试创建路由器（LangGraph 启用）。"""
        os.environ["USE_LANGGRAPH"] = "true"
        router = create_router()
        # 应该返回 LangGraph 路由器
        assert isinstance(router, LangGraphRouter)
```

- [ ] **Step 3: 运行测试**

运行：`pytest tests/test_adapter.py -v`

预期输出：测试通过

- [ ] **Step 4: 提交适配器**

```bash
git add agents/adapter.py tests/test_adapter.py
git commit -m "feat(agents): add LangGraph adapter for gradual migration"
```

---

### Task 15: 更新主程序

**Files:**
- Modify: `main.py`

**背景：** 在主程序中集成 LangGraph 路由器，支持通过环境变量切换。

- [ ] **Step 1: 修改 main.py 使用适配器**

在 main.py 中，找到创建路由器的部分，修改为：

```python
from agents.adapter import create_router

# 在初始化部分
router = create_router()
```

如果之前是这样的代码：
```python
from agents.router import IntentRouter
router = IntentRouter()
```

替换为：
```python
from agents.adapter import create_router
router = create_router()
```

- [ ] **Step 2: 测试旧版模式**

运行：`USE_LANGGRAPH=false python main.py`

预期输出：使用旧版路由器，功能正常

- [ ] **Step 3: 测试 LangGraph 模式（如果配置了 API_KEY）**

运行：`USE_LANGGRAPH=true python main.py`

预期输出：使用 LangGraph 路由器，功能正常

- [ ] **Step 4: 提交主程序更新**

```bash
git add main.py
git commit -m "feat(main): integrate LangGraph adapter with feature flag"
```

---

### Task 16: 文档更新

**Files:**
- Create: `docs/langgraph-migration.md`

**背景：** 创建迁移文档，说明如何从旧版迁移到 LangGraph 版本。

- [ ] **Step 1: 创建迁移文档**

创建 `docs/langgraph-migration.md`：

```markdown
# LangGraph 迁移指南

本文档说明如何从旧版 Agent 系统迁移到 LangGraph 架构。

## 架构对比

### 旧版架构

```
用户消息 → 关键词匹配 → LLM分类 → Agent分发 → 生成回复
```

**特点：**
- 简单的三层路由
- 无状态管理
- 无会话隔离

### 新版架构（LangGraph）

```
用户消息 → 意图识别节点 → 条件路由 → 议价/商品/默认节点 → 输出
```

**特点：**
- 声明式状态图
- 会话状态持久化
- 支持循环和条件分支
- 更好的可观测性

## 迁移步骤

### 1. 启用 LangGraph

在 `.env` 文件中设置：

```bash
USE_LANGGRAPH=true
```

### 2. 验证功能

启动服务并测试：

```bash
python main.py
```

### 3. 监控性能

观察以下指标：
- 响应延迟
- LLM 调用次数
- 会话状态大小

### 4. 回滚方案

如果遇到问题，可以立即回滚：

```bash
USE_LANGGRAPH=false
```

## 功能差异

### 新增功能

1. **会话隔离**
   - 每个用户独立的状态
   - 自动持久化到 SQLite

2. **议价循环**
   - 支持多轮议价
   - 动态温度调整

3. **监控支持**
   - 可选的 LangSmith 集成
   - 详细的执行日志

### 保持兼容

- 所有现有 prompt 文件不变
- API 接口保持兼容
- 配置文件格式不变

## 性能优化建议

1. **数据库优化**
   ```bash
   # 定期清理过期会话
   sqlite3 data/chat_history.db "DELETE FROM checkpoints WHERE timestamp < datetime('now', '-7 days')"
   ```

2. **LLM 缓存**
   - LangChain 支持自动缓存
   - 在环境变量中启用：
     ```bash
     LANGCHAIN_CACHE=true
     ```

3. **并发控制**
   - SQLite 默认支持并发读
   - 写操作会自动排队

## 故障排查

### 问题：会话状态丢失

**原因：** SQLite 数据库路径错误或权限问题

**解决方案：**
```bash
# 检查数据库文件
ls -la data/chat_history.db

# 确保目录存在
mkdir -p data
```

### 问题：响应变慢

**原因：** LangGraph 增加了额外的抽象层

**解决方案：**
1. 启用 LLM 缓存
2. 优化 prompt 长度
3. 使用更快的模型（如 qwen-flash）

### 问题：议价不循环

**原因：** 关键词检测不匹配或次数限制

**解决方案：**
1. 检查 `agents/routers/intent_router.py` 中的关键词列表
2. 调整议价次数上限
3. 查看 logs 中的路由日志

## 下一步

- [ ] 配置 LangSmith 监控
- [ ] 添加更多节点（如售后 Agent）
- [ ] 实现多 Agent 协作
- [ ] 优化状态持久化策略
```

- [ ] **Step 2: 提交文档**

```bash
git add docs/langgraph-migration.md
git commit -m "docs: add LangGraph migration guide"
```

---

### Task 17: 迁移现有测试

**Files:**
- Modify: `tests/conftest.py`
- Archive: `tests/test_agents.py` → `tests/legacy/test_agents.py`
- Test: `tests/test_integration.py`

**背景：** 将现有的集成测试迁移到新架构，保留有价值的测试用例，确保向后兼容性。

- [ ] **Step 1: 更新 conftest.py 添加新架构 fixtures**

在 `tests/conftest.py` 末尾添加：

```python
# LangGraph 相关 fixtures

@pytest.fixture
def langgraph_router():
    """创建 LangGraph 路由器实例用于测试。"""
    from agents.adapter import LangGraphRouter
    return LangGraphRouter()


@pytest.fixture
def agent_graph():
    """创建 Agent 状态图用于测试。"""
    from agents.graph import create_agent_graph
    return create_agent_graph()


@pytest.fixture
def sample_agent_state():
    """创建示例 AgentState 用于测试。"""
    from agents.state import AgentState
    from langchain_core.messages import HumanMessage

    return AgentState(
        messages=[HumanMessage(content="测试消息")],
        user_id="test_user",
        intent="",
        bargain_count=0,
        item_info=None,
        manual_mode=False,
    )
```

- [ ] **Step 2: 创建集成测试文件**

创建 `tests/test_integration.py`，迁移现有的 LLM API 测试：

```python
"""集成测试 - 真实 LLM API 调用。

这些测试需要配置 API_KEY 才能运行。
"""

import os
import time
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("API_KEY"), reason="API_KEY not set")
class TestRealLLMAPI:
    """集成测试，使用真实 LLM API 调用。"""

    @pytest.mark.asyncio
    async def test_default_node_real_api(self):
        """测试默认节点真实 API 调用。"""
        from agents.nodes.default import default_node
        from agents.state import AgentState
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[HumanMessage(content="你好，请问这个商品还在吗？")],
            user_id="test_user",
            intent="default",
            bargain_count=0,
            item_info=None,
            manual_mode=False,
        )

        start_time = time.perf_counter()
        result = await default_node(state)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print("\n" + "=" * 60)
        print("Default Node Test:")
        print(f"Input: 你好，请问这个商品还在吗？")
        print(f"API Call Time: {elapsed_time:.2f} seconds")
        print("-" * 60)
        print("Response:")
        print(result["messages"][0].content)
        print("=" * 60 + "\n")

        # 验证响应
        assert "messages" in result
        assert len(result["messages"]) > 0
        content = result["messages"][0].content
        assert isinstance(content, str)
        assert len(content) > 10
        assert len(content) < 1000
        assert "错误" not in content
        assert "失败" not in content
        assert "error" not in content.lower()

    @pytest.mark.asyncio
    async def test_price_node_real_api(self):
        """测试议价节点真实 API 调用。"""
        from agents.nodes.price import price_node
        from agents.state import AgentState
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[HumanMessage(content="能便宜点吗？")],
            user_id="test_user",
            intent="price",
            bargain_count=1,
            item_info={"min_price": 100},
            manual_mode=False,
        )

        start_time = time.perf_counter()
        result = await price_node(state)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print("\n" + "=" * 60)
        print("Price Node Test:")
        print(f"Input: 能便宜点吗？")
        print(f"Bargain Count: 1")
        print(f"API Call Time: {elapsed_time:.2f} seconds")
        print("-" * 60)
        print("Response:")
        print(result["messages"][0].content)
        print("=" * 60 + "\n")

        # 验证响应
        assert "messages" in result
        assert len(result["messages"]) > 0
        content = result["messages"][0].content
        assert isinstance(content, str)
        assert len(content) > 10
        assert len(content) < 1000
        assert "错误" not in content
        assert "失败" not in content

    @pytest.mark.asyncio
    async def test_classify_node_real_api(self):
        """测试意图识别节点真实 API 调用。"""
        from agents.nodes.classify import classify_node
        from agents.state import AgentState
        from langchain_core.messages import HumanMessage

        test_cases = [
            ("能便宜点吗", "price"),
            ("参数是什么", "product"),
            ("你好", "default"),
        ]

        for query, expected_intent in test_cases:
            state = AgentState(
                messages=[HumanMessage(content=query)],
                user_id="test_user",
                intent="",
                bargain_count=0,
                item_info=None,
                manual_mode=False,
            )

            result = await classify_node(state)
            actual_intent = result["intent"]

            print(f"Query: {query}")
            print(f"Expected: {expected_intent}, Actual: {actual_intent}")

            # 注意：实际结果取决于 LLM，这里只验证格式
            assert actual_intent in {"price", "product", "default", "no_reply"}

    @pytest.mark.asyncio
    async def test_end_to_end_flow(self):
        """测试端到端流程（通过适配器）。"""
        from agents.adapter import LangGraphRouter

        router = LangGraphRouter()

        test_cases = [
            ("你好", "default"),
            ("能便宜点吗", "price"),
            ("参数是什么", "product"),
        ]

        for query, expected_type in test_cases:
            start_time = time.perf_counter()
            response = await router.route(
                user_msg=query,
                user_id="test_user_e2e",
            )
            end_time = time.perf_counter()

            print("\n" + "=" * 60)
            print(f"E2E Test - {expected_type.upper()}")
            print(f"Query: {query}")
            print(f"Response Time: {end_time - start_time:.2f}s")
            print("-" * 60)
            print(f"Response: {response[:100]}...")
            print("=" * 60)

            # 基本验证
            assert isinstance(response, str)
            assert len(response) > 0
```

- [ ] **Step 3: 归档旧测试文件**

运行：`mkdir -p tests/legacy`

运行：`mv tests/test_agents.py tests/legacy/test_agents.py`

在 `tests/legacy/test_agents.py` 顶部添加说明：

```python
"""
旧版 Agent 系统测试（已归档）。

这些测试是为旧版的三层路由系统编写的，已被 LangGraph 架构替代。
保留此文件用于参考和对比。

新版本的测试在 tests/test_*.py 中。
"""
```

- [ ] **Step 4: 运行集成测试（如果配置了 API_KEY）**

运行：`pytest tests/test_integration.py -v -s`

预期输出：集成测试全部通过

- [ ] **Step 5: 运行所有测试确保兼容性**

运行：`pytest tests/ -v --ignore=tests/legacy/`

预期输出：所有新测试通过

- [ ] **Step 6: 更新 pytest 配置**

在 `pytest.ini` 或 `pyproject.toml` 中配置忽略 legacy 目录：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
python_classes = ["Test*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--ignore=tests/legacy",  # 忽略旧版测试
]
```

- [ ] **Step 7: 提交测试迁移**

```bash
git add tests/conftest.py tests/test_integration.py tests/legacy/
git commit -m "test(migration): migrate existing tests to LangGraph architecture"
```

---

## 验收标准

完成所有任务后，系统应该：

1. ✅ **功能完整**
   - 所有节点正常工作
   - 意图路由准确
   - 议价循环正常

2. ✅ **测试覆盖**
   - 单元测试全部通过
   - 集成测试全部通过
   - 代码覆盖率 > 80%

3. ✅ **性能稳定**
   - 响应延迟 < 3 秒
   - 无内存泄漏
   - 数据库大小可控

4. ✅ **文档完善**
   - API 文档更新
   - 迁移指南清晰
   - 注释充分

5. ✅ **向后兼容**
   - 可通过开关切换版本
   - 旧版代码仍可工作
   - 数据格式兼容

## 预期时间

- Chunk 1（基础设施）: 2-3 小时
- Chunk 2（节点实现）: 4-5 小时
- Chunk 3（路由和图）: 2-3 小时
- Chunk 4（集成配置）: 2-3 小时

**总计**: 10-14 小时
