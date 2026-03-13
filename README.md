# Goofish Customer Skill

基于 WebSocket + Playwright 双通道的闲鱼智能客服系统。通过 LLM 自动回复买家消息，支持智能议价、商品咨询和人工接管。

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| 语言 | Python 3.11+ | 全异步架构 (asyncio) |
| 消息通道 | websockets | WebSocket 主力通道，直连闲鱼 IMPaaS |
| 备用通道 | Playwright | 浏览器自动化备用通道 + Cookie 续期 |
| AI | OpenAI SDK | 兼容 OpenAI API 格式，默认通义千问 qwen-max |
| HTTP | httpx | 异步 HTTP 客户端，用于 Token 获取和商品查询 |
| 存储 | SQLite | 对话历史、议价计数、商品缓存 |
| 配置 | PyYAML + python-dotenv | YAML 配置 + .env 环境变量 |
| 日志 | loguru | 终端输出 + 按天轮转文件日志 |
| 代码质量 | Ruff + Lefthook | 自动格式化 + Git Hooks |

## 项目结构

```
goofish-customer-skill/
├── main.py                     # 应用入口
├── core/                       # 消息通道
│   ├── channel.py              #   抽象基类 (MessageChannel)
│   ├── websocket_channel.py    #   WebSocket 通道（主力）
│   └── browser_channel.py      #   Playwright 通道（备用）
├── services/                   # 闲鱼平台服务
│   ├── xianyu_api.py           #   HTTP API 封装
│   └── xianyu_utils.py         #   签名、设备ID、MessagePack 解码
├── auth/                       # 认证管理
│   ├── cookie_manager.py       #   Cookie 存储与加载
│   ├── cookie_refresher.py     #   Playwright Cookie 续期
│   └── token_manager.py        #   WebSocket Token 刷新
├── agents/                     # AI Agent 体系
│   ├── router.py               #   三级意图路由
│   ├── classify_agent.py       #   意图分类 Agent
│   ├── price_agent.py          #   智能议价 Agent
│   ├── product_agent.py        #   商品咨询 Agent
│   └── default_agent.py        #   默认回复 Agent + BaseAgent 基类
├── storage/                    # 数据存储
│   └── context_manager.py      #   SQLite 对话历史管理
├── config/                     # 配置文件
│   ├── settings.yaml           #   全局设置
│   ├── products.yaml           #   商品补充信息
│   └── prompts/                #   Agent 话术模板
├── data/                       # 运行时数据（自动生成）
│   ├── cookies.json
│   ├── chat_history.db
│   └── logs/
├── tests/                      # 测试
├── requirements.txt
├── .env.example
└── .gitignore
```

## 快速开始

### 1. 安装依赖

**先安装 uv（推荐）：**

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 Homebrew
brew install uv

# 或使用 pip
pip install uv
```

**方式一：一键安装（推荐）**

```bash
make install
```

**方式二：手动安装**

```bash
uv pip sync requirements.txt
uv run playwright install chromium
uv run lefthook install
```

**依赖管理说明：**
- `pyproject.toml` - 依赖声明源（添加/删除依赖时修改这里）
- `requirements.txt` - 从 `pyproject.toml` 自动生成（运行 `make sync-requirements`）
- 使用 uv 管理依赖，比 pip 快 10-100 倍
- Makefile 使用 `uv run` 自动管理虚拟环境
- Lefthook 会在每次 `git commit` 时自动运行 `ruff format` 和 `ruff check`
- 代码格式化问题会自动修复，严重错误会阻止提交
- 手动执行：`make format` 或 `make check`

**更新依赖流程：**
```bash
# 1. 在 pyproject.toml 中添加/修改依赖
# 2. 生成新的 requirements.txt
make sync-requirements

# 3. 安装更新后的依赖
make install-deps
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入必需配置：

| 变量 | 必需 | 说明 |
|------|------|------|
| `API_KEY` | 是 | 通义千问 API Key（或其他 OpenAI 兼容 API Key） |
| `COOKIES_STR` | 是 | 闲鱼网页版 Cookie 字符串 |

### 3. 获取 Cookie

**方式一：浏览器扫码登录（推荐）**

```bash
python main.py --login
```

会打开 Chromium 浏览器，扫码登录后 Cookie 自动保存到 `data/cookies.json`。

**方式二：手动复制**

从浏览器开发者工具复制闲鱼网页版的 Cookie 字符串，粘贴到 `.env` 的 `COOKIES_STR` 中。

### 4. 启动服务

```bash
python main.py
```

## 配置说明

### 环境变量 (.env)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `API_KEY` | - | LLM API Key |
| `COOKIES_STR` | - | 闲鱼 Cookie |
| `MODEL_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | LLM API 地址 |
| `MODEL_NAME` | `qwen-max` | LLM 模型名称 |
| `TOGGLE_KEYWORDS` | `#manual` | 人工接管触发词 |
| `SIMULATE_HUMAN_TYPING` | `False` | 是否模拟打字延迟 |
| `LOG_LEVEL` | `DEBUG` | 日志级别 |
| `HEARTBEAT_INTERVAL` | `15` | WebSocket 心跳间隔（秒） |
| `TOKEN_REFRESH_INTERVAL` | `3600` | Token 刷新间隔（秒） |
| `MANUAL_MODE_TIMEOUT` | `3600` | 人工接管超时（秒） |
| `MESSAGE_EXPIRE_TIME` | `300000` | 消息过期时间（毫秒） |

### 商品信息 (config/products.yaml)

为商品补充 API 拿不到的信息（最低价、卖点等）：

```yaml
products:
  - item_id: "123456789"
    min_price: 4000
    selling_points: "95新，无磕碰"
    keywords: ["iphone", "苹果15"]
    notes: "不包邮偏远地区"
```

### 话术模板 (config/prompts/)

每个 Agent 的 system prompt 以 Markdown 文件存储，可直接编辑定制回复风格：

- `global_rules.md` — 全局规则
- `classify_prompt.md` — 意图分类
- `price_prompt.md` — 议价策略（支持 `{min_price}` `{bargain_count}` 占位符）
- `product_prompt.md` — 商品咨询
- `default_prompt.md` — 默认回复

## 核心功能

### 双通道消息机制

- **WebSocket 通道（主力）**：直连 `wss://wss-goofish.dingtalk.com/`，低延迟实时收发
- **Playwright 通道（备用）**：WebSocket 不可用时自动降级为浏览器模式
- **自动切换**：WebSocket 重连 3 次失败后切换 Playwright，Cookie 续期成功后切回

### 三级意图路由

```
买家消息 → 关键词匹配（零成本） → LLM 分类 → Agent 生成回复
```

1. **关键词匹配**：商品关键词优先于议价关键词，无 LLM 调用
2. **LLM 分类**：ClassifyAgent 判定 price / product / default / no_reply
3. **Agent 回复**：对应 Agent 结合商品信息 + 对话历史生成回复

### 智能议价

PriceAgent 使用动态 temperature 策略：

- 第 1 次议价 (t=0.3)：坚持原价
- 第 2 次 (t=0.45)：小幅让步
- 第 3 次 (t=0.6)：接近底价
- 3 次以上 (t=0.9)：坚持底价，委婉拒绝

### 人工接管

卖家在对话中发送 `#manual`（可配置），暂停该会话的自动回复。再次发送恢复，或 1 小时后自动恢复。

### 安全过滤

所有 LLM 回复自动检测敏感词（微信/QQ/支付宝/银行卡/线下交易），命中则替换为平台沟通提醒。

## 运行测试

```bash
python -m pytest tests/ -v
```

## 后续规划

- AftersaleAgent：独立售后 Agent + 售后事件邮件通知
- Rust/Tauri 重写为桌面应用
- 多账号支持
- 图片/富媒体消息支持

## License

Private project.
