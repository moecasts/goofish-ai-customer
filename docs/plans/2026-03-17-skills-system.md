# Skills System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace hardcoded agent nodes with a file-driven skills system where each skill is a directory containing a `skill.md` file (YAML frontmatter + prompt body), consistent with Claude Code's skills format.

**Architecture:** A `SkillRegistry` scans `config/skills/*/skill.md` at startup and injects skill metadata into the classify prompt dynamically. A single `skill_executor` node replaces all three response nodes (`price`, `product`, `default`), handling template variable injection and `state_hooks`-driven stateful logic. The LangGraph graph routes all non-`no_reply` intents through `skill_executor`.

**Tech Stack:** Python 3.11+, LangGraph, LangChain Core, PyYAML (already in deps), uv for running commands

---

## Task 1: Create skill directories and migrate prompt files

**Files:**
- Create: `config/skills/price/skill.md`
- Create: `config/skills/product/skill.md`
- Create: `config/skills/default/skill.md`
- Delete (after migration): `config/prompts/price_prompt.md`, `config/prompts/product_prompt.md`, `config/prompts/default_prompt.md`

**Step 1: Create price skill**

```bash
mkdir -p config/skills/price
```

Create `config/skills/price/skill.md`:

```markdown
---
name: price
description: 处理买家议价、砍价、询价请求
state_hooks:
  - bargain_count
  - min_price
---

你是闲鱼卖家的议价助手。

## 议价策略
- 第 1 次议价：坚持原价，强调商品价值
- 第 2 次议价：可小幅让步（不超过原价的 5%）
- 第 3 次议价：给出接近底价的价格
- 超过 3 次：坚持底价，委婉拒绝继续砍价

## 注意
- 不能低于最低价 {min_price}
- 语气友好但坚定
- 当前议价次数: {bargain_count}

## 重要
只回复用户最后一条消息，忽略之前的对话历史。直接生成回复，不要重复或总结之前的对话。
```

**Step 2: Create product skill**

```bash
mkdir -p config/skills/product
```

Create `config/skills/product/skill.md`:

```markdown
---
name: product
description: 处理商品详情、参数、规格、使用方法等咨询请求
---

你是闲鱼卖家的商品咨询助手。根据商品信息回答买家的问题。

## 注意
- 如实描述商品状况，不夸大
- 不确定的信息不要编造
- 引导买家查看商品详情页
```

**Step 3: Create default skill**

```bash
mkdir -p config/skills/default
```

Create `config/skills/default/skill.md`:

```markdown
---
name: default
description: 处理打招呼、闲聊及其他未分类消息
---

你是闲鱼卖家的客服助手。友好回复买家的各类消息。

## 注意
- 热情但不过分
- 引导买家关注商品
- 遇到售后问题，告知买家稍后卖家本人会处理
```

**Step 4: Commit the new skill files (do NOT delete old prompts yet)**

```bash
git add config/skills/
git commit -m "feat(config): add skills directory with price, product, default skills"
```

---

## Task 2: Implement SkillRegistry

**Files:**
- Create: `agents/skill_registry.py`
- Create: `tests/test_skill_registry.py`

**Step 1: Write failing tests first**

Create `tests/test_skill_registry.py`:

```python
"""测试 SkillRegistry。"""

import pytest
from pathlib import Path
from agents.skill_registry import Skill, SkillRegistry


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """创建临时 skills 目录结构。"""
    # price skill with state_hooks
    price_dir = tmp_path / "price"
    price_dir.mkdir()
    (price_dir / "skill.md").write_text(
        "---\nname: price\ndescription: 处理议价\nstate_hooks:\n  - bargain_count\n  - min_price\n---\n\n议价 prompt {bargain_count} {min_price}"
    )

    # product skill without state_hooks
    product_dir = tmp_path / "product"
    product_dir.mkdir()
    (product_dir / "skill.md").write_text(
        "---\nname: product\ndescription: 处理商品咨询\n---\n\n商品咨询 prompt"
    )

    # invalid skill missing description
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "skill.md").write_text(
        "---\nname: bad\n---\n\n没有 description 的 skill"
    )

    return tmp_path


def test_load_skills_basic(skills_dir: Path):
    """测试基本加载：price 和 product 成功，bad 被跳过。"""
    registry = SkillRegistry(skills_dir)
    skills = registry.list_skills()
    names = {s.name for s in skills}
    assert "price" in names
    assert "product" in names
    assert "bad" not in names


def test_skill_state_hooks(skills_dir: Path):
    """测试 state_hooks 正确解析。"""
    registry = SkillRegistry(skills_dir)
    price = registry.get_skill("price")
    assert price.state_hooks == ["bargain_count", "min_price"]


def test_skill_no_state_hooks(skills_dir: Path):
    """测试没有 state_hooks 时默认为空列表。"""
    registry = SkillRegistry(skills_dir)
    product = registry.get_skill("product")
    assert product.state_hooks == []


def test_skill_prompt_body(skills_dir: Path):
    """测试 prompt 正文（frontmatter 之后的内容）正确提取。"""
    registry = SkillRegistry(skills_dir)
    product = registry.get_skill("product")
    assert "商品咨询 prompt" in product.prompt
    assert "---" not in product.prompt


def test_skill_dir_path(skills_dir: Path):
    """测试 skill_dir 指向正确的目录。"""
    registry = SkillRegistry(skills_dir)
    price = registry.get_skill("price")
    assert price.skill_dir == skills_dir / "price"


def test_get_unknown_skill_returns_none(skills_dir: Path):
    """测试获取不存在的 skill 返回 None。"""
    registry = SkillRegistry(skills_dir)
    result = registry.get_skill("nonexistent")
    assert result is None


def test_build_classify_context(skills_dir: Path):
    """测试生成 classify prompt 注入内容。"""
    registry = SkillRegistry(skills_dir)
    context = registry.build_classify_context()
    assert "price" in context
    assert "处理议价" in context
    assert "product" in context
    assert "处理商品咨询" in context


def test_skills_dir_not_exists():
    """测试 skills 目录不存在时抛出错误。"""
    with pytest.raises(FileNotFoundError, match="skills"):
        SkillRegistry(Path("/nonexistent/path/skills"))
```

**Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/test_skill_registry.py -v 2>&1 | head -30
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'agents.skill_registry'`

**Step 3: Implement SkillRegistry**

Create `agents/skill_registry.py`:

```python
"""Skills 注册表：扫描并管理 config/skills/ 目录下的所有 skill。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
from loguru import logger


@dataclass
class Skill:
    """单个 skill 的数据结构。"""

    name: str
    description: str
    prompt: str
    state_hooks: list[str] = field(default_factory=list)
    skill_dir: Path = field(default_factory=Path)


class SkillRegistry:
    """扫描 skills 目录，管理所有已注册 skill。"""

    def __init__(self, skills_dir: Path):
        if not skills_dir.exists():
            raise FileNotFoundError(f"skills 目录不存在: {skills_dir}")
        self._skills: dict[str, Skill] = {}
        self._load_skills(skills_dir)

    def _load_skills(self, skills_dir: Path) -> None:
        """扫描 skills_dir/*/skill.md 并加载所有有效 skill。"""
        for skill_md in sorted(skills_dir.glob("*/skill.md")):
            skill = self._parse_skill(skill_md)
            if skill:
                self._skills[skill.name] = skill
                logger.info(f"已加载 skill: {skill.name}")

    def _parse_skill(self, skill_md: Path) -> Optional[Skill]:
        """解析单个 skill.md 文件，返回 Skill 或 None（解析失败时）。"""
        try:
            content = skill_md.read_text(encoding="utf-8")
            frontmatter, prompt = self._split_frontmatter(content)
            if frontmatter is None:
                logger.warning(f"skill 缺少 frontmatter: {skill_md}")
                return None

            meta = yaml.safe_load(frontmatter)
            if not meta.get("name") or not meta.get("description"):
                logger.warning(f"skill 缺少 name 或 description，跳过: {skill_md}")
                return None

            return Skill(
                name=meta["name"],
                description=meta["description"],
                prompt=prompt.strip(),
                state_hooks=meta.get("state_hooks") or [],
                skill_dir=skill_md.parent,
            )
        except Exception as e:
            logger.warning(f"解析 skill 失败，跳过 {skill_md}: {e}")
            return None

    def _split_frontmatter(self, content: str) -> tuple[Optional[str], str]:
        """分割 YAML frontmatter 和 Markdown 正文。"""
        if not content.startswith("---"):
            return None, content
        parts = content.split("---", 2)
        if len(parts) < 3:
            return None, content
        return parts[1], parts[2]

    def get_skill(self, name: str) -> Optional[Skill]:
        """按 name 获取 skill，不存在时返回 None。"""
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        """返回所有已注册 skill 列表。"""
        return list(self._skills.values())

    def build_classify_context(self) -> str:
        """生成注入 classify prompt 的 skill 列表描述。"""
        lines = [f"- {s.name}: {s.description}" for s in self._skills.values()]
        return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_skill_registry.py -v
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add agents/skill_registry.py tests/test_skill_registry.py
git commit -m "feat(agents): implement SkillRegistry for file-driven skill loading"
```

---

## Task 3: Implement skill_executor node

**Files:**
- Create: `agents/nodes/skill_executor.py`
- Create: `tests/test_skill_executor.py`

**Step 1: Write failing tests**

Create `tests/test_skill_executor.py`:

```python
"""测试 skill_executor 节点。"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents.nodes.skill_executor import make_skill_executor
from agents.skill_registry import Skill, SkillRegistry
from agents.state import AgentState


def make_mock_registry(skill: Skill) -> MagicMock:
    """创建返回指定 skill 的 mock registry。"""
    registry = MagicMock(spec=SkillRegistry)
    registry.get_skill.return_value = skill
    return registry


@pytest.fixture
def price_skill() -> Skill:
    return Skill(
        name="price",
        description="处理议价",
        prompt="议价 prompt，底价 {min_price}，次数 {bargain_count}",
        state_hooks=["bargain_count", "min_price"],
        skill_dir=Path("/tmp/price"),
    )


@pytest.fixture
def product_skill() -> Skill:
    return Skill(
        name="product",
        description="处理商品咨询",
        prompt="商品咨询 prompt",
        state_hooks=[],
        skill_dir=Path("/tmp/product"),
    )


@pytest.mark.asyncio
async def test_executor_injects_state_hooks(price_skill):
    """测试 state_hooks 变量被正确注入 prompt。"""
    registry = make_mock_registry(price_skill)
    mock_response = MagicMock()
    mock_response.content = "好的，这个价格可以"
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    state = AgentState(
        messages=[HumanMessage(content="能便宜点吗")],
        user_id="u1",
        intent="price",
        bargain_count=1,
        item_info={"min_price": "100", "product_name": "手机"},
        manual_mode=False,
    )

    with patch("agents.nodes.skill_executor.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor.check_safety", side_effect=lambda x: x):
        executor = make_skill_executor(registry)
        result = await executor(state)

    # 验证 LLM 被调用时 prompt 包含注入的变量值
    call_args = mock_llm.invoke.call_args[0][0]
    system_msg = call_args[0]
    assert isinstance(system_msg, SystemMessage)
    assert "100" in system_msg.content    # min_price 注入
    assert "1" in system_msg.content      # bargain_count 注入


@pytest.mark.asyncio
async def test_executor_increments_bargain_count(price_skill):
    """测试 price skill 的 bargain_count 在执行后 +1。"""
    registry = make_mock_registry(price_skill)
    mock_response = MagicMock()
    mock_response.content = "价格不能再低了"
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    state = AgentState(
        messages=[HumanMessage(content="再便宜一点")],
        user_id="u1",
        intent="price",
        bargain_count=2,
        item_info={"min_price": "80"},
        manual_mode=False,
    )

    with patch("agents.nodes.skill_executor.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor.check_safety", side_effect=lambda x: x):
        executor = make_skill_executor(registry)
        result = await executor(state)

    assert result["bargain_count"] == 3


@pytest.mark.asyncio
async def test_executor_no_bargain_count_for_product(product_skill):
    """测试 product skill 执行后不修改 bargain_count。"""
    registry = make_mock_registry(product_skill)
    mock_response = MagicMock()
    mock_response.content = "这款商品..."
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    state = AgentState(
        messages=[HumanMessage(content="这个商品多大？")],
        user_id="u1",
        intent="product",
        bargain_count=0,
        item_info={},
        manual_mode=False,
    )

    with patch("agents.nodes.skill_executor.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor.check_safety", side_effect=lambda x: x):
        executor = make_skill_executor(registry)
        result = await executor(state)

    assert "bargain_count" not in result


@pytest.mark.asyncio
async def test_executor_applies_safety_filter(product_skill):
    """测试 LLM 输出经过 check_safety 过滤。"""
    registry = make_mock_registry(product_skill)
    mock_response = MagicMock()
    mock_response.content = "加我微信"
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    state = AgentState(
        messages=[HumanMessage(content="怎么联系你？")],
        user_id="u1",
        intent="product",
        bargain_count=0,
        item_info={},
        manual_mode=False,
    )

    with patch("agents.nodes.skill_executor.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor.check_safety", return_value="[安全提醒]"):
        executor = make_skill_executor(registry)
        result = await executor(state)

    assert result["messages"][-1].content == "[安全提醒]"


@pytest.mark.asyncio
async def test_executor_unknown_skill_falls_back_to_default(product_skill):
    """测试未知 skill name 时 registry 返回 None，executor 使用 fallback 回复。"""
    registry = MagicMock(spec=SkillRegistry)
    registry.get_skill.return_value = None  # 未知 skill

    state = AgentState(
        messages=[HumanMessage(content="随便说点什么")],
        user_id="u1",
        intent="unknown_skill",
        bargain_count=0,
        item_info={},
        manual_mode=False,
    )

    executor = make_skill_executor(registry)
    result = await executor(state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)


@pytest.mark.asyncio
async def test_executor_llm_error_returns_fallback(price_skill):
    """测试 LLM 调用失败时返回 fallback 回复。"""
    registry = make_mock_registry(price_skill)
    mock_llm = AsyncMock()
    mock_llm.invoke.side_effect = Exception("LLM 挂了")

    state = AgentState(
        messages=[HumanMessage(content="能便宜吗")],
        user_id="u1",
        intent="price",
        bargain_count=0,
        item_info={"min_price": "50"},
        manual_mode=False,
    )

    with patch("agents.nodes.skill_executor.LLMClient", return_value=mock_llm), \
         patch("agents.nodes.skill_executor.check_safety", side_effect=lambda x: x):
        executor = make_skill_executor(registry)
        result = await executor(state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
```

**Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/test_skill_executor.py -v 2>&1 | head -20
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'agents.nodes.skill_executor'`

**Step 3: Implement skill_executor**

Create `agents/nodes/skill_executor.py`:

```python
"""通用 skill 执行节点。"""

from typing import Callable
from langchain_core.messages import AIMessage, SystemMessage
from agents.state import AgentState
from agents.skill_registry import SkillRegistry
from services.llm_client import LLMClient
from services.tools import check_safety
from loguru import logger


def make_skill_executor(registry: SkillRegistry) -> Callable:
    """工厂函数：创建绑定了 registry 的 skill_executor 节点函数。"""

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
        prompt = skill.prompt
        for key, value in variables.items():
            prompt = prompt.replace(f"{{{key}}}", value)

        # 未替换的占位符记录警告
        import re
        remaining = re.findall(r"\{(\w+)\}", prompt)
        if remaining:
            logger.warning(f"skill '{intent}' prompt 中有未替换的变量: {remaining}")

        # 3. 调用 LLM
        llm_client = LLMClient()
        messages = [SystemMessage(content=prompt), *state["messages"]]

        try:
            response = await llm_client.invoke(messages, allow_empty=False)
            safe_content = check_safety(response.content)
            logger.info(f"skill_executor [{intent}] 执行成功，回复长度={len(safe_content)}")

            # 4. 构建返回 state
            updates: AgentState = {"messages": [AIMessage(content=safe_content)]}

            # 5. state_hooks 写回：bargain_count +1
            if "bargain_count" in skill.state_hooks:
                try:
                    updates["bargain_count"] = int(state.get("bargain_count", 0)) + 1
                except (TypeError, ValueError) as e:
                    logger.error(f"bargain_count 写回失败，保持原值: {e}")

            return updates

        except Exception as e:
            logger.error(f"skill_executor [{intent}] 执行失败: {e}")
            return {"messages": [AIMessage(content="抱歉，处理请求时出错了")]}

    return skill_executor_node
```

**Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_skill_executor.py -v
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add agents/nodes/skill_executor.py tests/test_skill_executor.py
git commit -m "feat(agents): implement generic skill_executor node"
```

---

## Task 4: Update classify node to use dynamic skill list

**Files:**
- Modify: `agents/nodes/classify.py`
- Modify: `config/prompts/classify_prompt.md`
- Modify: `tests/test_classify_node.py`

**Step 1: Update classify_prompt.md**

Replace `config/prompts/classify_prompt.md` content with:

```markdown
你是一个意图分类器。根据买家消息，判断其意图类别。

可用的 skill 列表：
{skills}
- no_reply: 无需回复（如表情、图片、系统消息）

仅输出以下 skill name 之一，不要输出其他内容。
```

**Step 2: Update classify node**

Replace `agents/nodes/classify.py` with:

```python
"""意图识别节点。"""

from langchain_core.messages import SystemMessage
from agents.state import AgentState
from agents.skill_registry import SkillRegistry
from services.llm_client import LLMClient
from loguru import logger


def make_classify_node(registry: SkillRegistry):
    """工厂函数：创建绑定了 registry 的 classify 节点函数。"""

    async def classify_node(state: AgentState) -> AgentState:
        """意图识别节点：动态注入 skill 列表，LLM 返回 skill name。"""
        llm_client = LLMClient()

        try:
            with open("config/prompts/classify_prompt.md", "r", encoding="utf-8") as f:
                prompt_template = f.read()
        except FileNotFoundError:
            logger.error("classify_prompt.md 文件未找到")
            return {"intent": "default"}

        # 动态注入 skill 列表
        skills_context = registry.build_classify_context()
        system_prompt = prompt_template.replace("{skills}", skills_context)

        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]

        # 合法的 intent = 所有 skill name + no_reply
        valid_intents = {s.name for s in registry.list_skills()} | {"no_reply"}

        try:
            response = await llm_client.invoke(messages)
            intent = response.content.strip().lower()
            if intent not in valid_intents:
                logger.warning(f"classify 返回未知 intent '{intent}'，降级为 default")
                intent = "default"
            logger.info(f"意图识别结果: {intent}")
            return {"intent": intent}
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            return {"intent": "default"}

    return classify_node
```

**Step 3: Update test_classify_node.py**

Replace `tests/test_classify_node.py` with:

```python
"""测试意图识别节点。"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from agents.nodes.classify import make_classify_node
from agents.skill_registry import Skill, SkillRegistry
from agents.state import AgentState
from langchain_core.messages import HumanMessage


def make_registry_with_skills(*names: str) -> MagicMock:
    """创建包含指定 skill names 的 mock registry。"""
    registry = MagicMock(spec=SkillRegistry)
    skills = [Skill(name=n, description=f"{n} 描述", prompt="", skill_dir=MagicMock()) for n in names]
    registry.list_skills.return_value = skills
    registry.build_classify_context.return_value = "\n".join(
        f"- {s.name}: {s.description}" for s in skills
    )
    return registry


@pytest.mark.asyncio
async def test_classify_node_valid_intent():
    """测试识别有效意图。"""
    registry = make_registry_with_skills("price", "product", "default")
    mock_response = MagicMock()
    mock_response.content = "price"
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    with patch("agents.nodes.classify.LLMClient", return_value=mock_llm), \
         patch("builtins.open", mock_open(read_data="分类 prompt\n{skills}")):
        node = make_classify_node(registry)
        state = AgentState(messages=[HumanMessage(content="能便宜点吗")])
        result = await node(state)

    assert result["intent"] == "price"


@pytest.mark.asyncio
async def test_classify_node_invalid_intent_falls_back_to_default():
    """测试未知 intent 降级为 default。"""
    registry = make_registry_with_skills("price", "product", "default")
    mock_response = MagicMock()
    mock_response.content = "totally_unknown"
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    with patch("agents.nodes.classify.LLMClient", return_value=mock_llm), \
         patch("builtins.open", mock_open(read_data="分类 prompt\n{skills}")):
        node = make_classify_node(registry)
        state = AgentState(messages=[HumanMessage(content="随便")])
        result = await node(state)

    assert result["intent"] == "default"


@pytest.mark.asyncio
async def test_classify_node_no_reply_is_valid():
    """测试 no_reply 是合法 intent（内置保留值）。"""
    registry = make_registry_with_skills("price", "product", "default")
    mock_response = MagicMock()
    mock_response.content = "no_reply"
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    with patch("agents.nodes.classify.LLMClient", return_value=mock_llm), \
         patch("builtins.open", mock_open(read_data="分类 prompt\n{skills}")):
        node = make_classify_node(registry)
        state = AgentState(messages=[HumanMessage(content="😊")])
        result = await node(state)

    assert result["intent"] == "no_reply"


@pytest.mark.asyncio
async def test_classify_node_file_not_found():
    """测试 prompt 文件不存在时降级为 default。"""
    registry = make_registry_with_skills("price", "product", "default")
    with patch("builtins.open", side_effect=FileNotFoundError):
        node = make_classify_node(registry)
        state = AgentState(messages=[HumanMessage(content="你好")])
        result = await node(state)

    assert result["intent"] == "default"


@pytest.mark.asyncio
async def test_classify_node_llm_error():
    """测试 LLM 调用失败时降级为 default。"""
    registry = make_registry_with_skills("price", "product", "default")
    mock_llm = AsyncMock()
    mock_llm.invoke.side_effect = Exception("LLM 挂了")

    with patch("agents.nodes.classify.LLMClient", return_value=mock_llm), \
         patch("builtins.open", mock_open(read_data="分类 prompt\n{skills}")):
        node = make_classify_node(registry)
        state = AgentState(messages=[HumanMessage(content="你好")])
        result = await node(state)

    assert result["intent"] == "default"
```

**Step 4: Run updated tests**

```bash
uv run python -m pytest tests/test_classify_node.py -v
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add agents/nodes/classify.py config/prompts/classify_prompt.md tests/test_classify_node.py
git commit -m "feat(agents): update classify node to use dynamic skill registry"
```

---

## Task 5: Rewire LangGraph graph

**Files:**
- Modify: `agents/graph.py`
- Modify: `agents/nodes/__init__.py`
- Modify: `tests/test_graph.py`

**Step 1: Update agents/nodes/__init__.py**

Replace content:

```python
"""Agent 节点模块。"""

from agents.nodes.classify import make_classify_node
from agents.nodes.skill_executor import make_skill_executor

__all__ = [
    "make_classify_node",
    "make_skill_executor",
]
```

**Step 2: Update graph.py**

Replace `create_agent_graph()` and `route_intent()` in `agents/graph.py`:

```python
"""Agent 状态图构建。"""

import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from agents.state import AgentState
from agents.nodes import make_classify_node, make_skill_executor
from agents.skill_registry import SkillRegistry
from services.llm_client import LLMClient

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
    from pathlib import Path

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
```

Keep the `LangGraphRouter` class but update its `__init__` to pass `skills_dir`:

```python
class LangGraphRouter:
    def __init__(self, skills_dir: str = _SKILLS_DIR_DEFAULT):
        self.graph = create_agent_graph(skills_dir)
        self.llm_client = LLMClient()
        logger.info("LangGraphRouter 初始化完成")
```

**Step 3: Run existing graph tests to see what breaks**

```bash
uv run python -m pytest tests/test_graph.py tests/test_intent_router.py -v 2>&1 | head -50
```

**Step 4: Fix test_graph.py and test_intent_router.py**

Update `tests/test_graph.py` — replace tests that reference old nodes (`price_node`, `product_node`, `default_node`) with skill-aware equivalents:

```python
"""测试 Agent 状态图。"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from agents.graph import create_agent_graph, route_intent, LangGraphRouter
from agents.state import AgentState


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """创建临时 skills 目录供 graph 测试使用。"""
    for name, desc in [("price", "议价"), ("product", "商品咨询"), ("default", "默认回复")]:
        d = tmp_path / name
        d.mkdir()
        hooks = "\nstate_hooks:\n  - bargain_count\n  - min_price" if name == "price" else ""
        (d / "skill.md").write_text(
            f"---\nname: {name}\ndescription: {desc}{hooks}\n---\n\n{name} prompt"
        )
    return tmp_path


def test_create_agent_graph(skills_dir):
    """测试状态图可以成功创建。"""
    graph = create_agent_graph(str(skills_dir))
    assert graph is not None


def test_graph_structure(skills_dir):
    """测试状态图包含必要节点。"""
    graph = create_agent_graph(str(skills_dir))
    assert graph is not None


def test_route_intent_no_reply():
    """测试 no_reply 路由到 no_reply。"""
    state = AgentState(messages=[], user_id="u", intent="no_reply",
                       bargain_count=0, item_info=None, manual_mode=False)
    assert route_intent(state) == "no_reply"


def test_route_intent_price():
    """测试 price intent 路由到 skill。"""
    state = AgentState(messages=[], user_id="u", intent="price",
                       bargain_count=0, item_info=None, manual_mode=False)
    assert route_intent(state) == "skill"


def test_route_intent_default():
    """测试 default intent 路由到 skill。"""
    state = AgentState(messages=[], user_id="u", intent="default",
                       bargain_count=0, item_info=None, manual_mode=False)
    assert route_intent(state) == "skill"


def test_route_intent_missing():
    """测试 intent 为空时路由到 skill（默认 default）。"""
    state = AgentState(messages=[], user_id="u", intent="",
                       bargain_count=0, item_info=None, manual_mode=False)
    assert route_intent(state) == "skill"
```

Update `tests/test_intent_router.py` — remove tests for `check_bargain_continue` (this function no longer exists) and keep only `route_intent` tests (already covered above, so the file can be deleted or reduced):

```python
"""测试意图路由函数。"""

import pytest
from agents.graph import route_intent
from agents.state import AgentState


def make_state(intent: str) -> AgentState:
    return AgentState(messages=[], user_id="u", intent=intent,
                      bargain_count=0, item_info=None, manual_mode=False)


def test_route_intent_price():
    assert route_intent(make_state("price")) == "skill"

def test_route_intent_product():
    assert route_intent(make_state("product")) == "skill"

def test_route_intent_default():
    assert route_intent(make_state("default")) == "skill"

def test_route_intent_no_reply():
    assert route_intent(make_state("no_reply")) == "no_reply"

def test_route_intent_missing():
    assert route_intent(make_state("")) == "skill"
```

**Step 5: Run all affected tests**

```bash
uv run python -m pytest tests/test_graph.py tests/test_intent_router.py -v
```

Expected: All tests PASS.

**Step 6: Commit**

```bash
git add agents/graph.py agents/nodes/__init__.py tests/test_graph.py tests/test_intent_router.py
git commit -m "feat(agents): rewire graph to use SkillRegistry and skill_executor"
```

---

## Task 6: Delete old node files and prompt files

**Files:**
- Delete: `agents/nodes/classify.py` (old version — already replaced in Task 4)
- Delete: `agents/nodes/price.py`
- Delete: `agents/nodes/product.py`
- Delete: `agents/nodes/default.py`
- Delete: `config/prompts/price_prompt.md`
- Delete: `config/prompts/product_prompt.md`
- Delete: `config/prompts/default_prompt.md`
- Delete: `tests/test_price_node.py`
- Delete: `tests/test_product_node.py`
- Delete: `tests/test_default_node.py`

**Step 1: Run full test suite before deletion to confirm green**

```bash
uv run python -m pytest -v 2>&1 | tail -20
```

Expected: All skill/graph/classify tests PASS. Old node tests will still pass (they still import the old files). Note any failures.

**Step 2: Delete old node Python files**

```bash
rm agents/nodes/price.py agents/nodes/product.py agents/nodes/default.py
rm config/prompts/price_prompt.md config/prompts/product_prompt.md config/prompts/default_prompt.md
rm tests/test_price_node.py tests/test_product_node.py tests/test_default_node.py
```

**Step 3: Run test suite again to confirm nothing broke**

```bash
uv run python -m pytest -v 2>&1 | tail -20
```

Expected: All remaining tests PASS. No import errors.

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor(agents): remove legacy node files, migrate to skills system"
```

---

## Task 7: Final verification

**Step 1: Run full test suite**

```bash
uv run python -m pytest -v
```

Expected: All tests PASS, no import errors, no deprecation warnings about old nodes.

**Step 2: Verify skill loading works end-to-end**

```bash
uv run python -c "
from pathlib import Path
from agents.skill_registry import SkillRegistry
r = SkillRegistry(Path('config/skills'))
for s in r.list_skills():
    print(f'{s.name}: hooks={s.state_hooks}')
print('---')
print(r.build_classify_context())
"
```

Expected output:
```
price: hooks=['bargain_count', 'min_price']
product: hooks=[]
default: hooks=[]
---
- price: 处理买家议价、砍价、询价请求
- product: 处理商品详情、参数、规格、使用方法等咨询请求
- default: 处理打招呼、闲聊及其他未分类消息
```

**Step 3: Final commit tag**

```bash
git tag v-skills-system
```
