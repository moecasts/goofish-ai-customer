# 闲鱼智能客服实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现一个基于 WebSocket + Playwright 双通道的闲鱼自动客服系统，支持智能议价、商品咨询和人工接管。

**Architecture:** WebSocket 直连闲鱼 IMPaaS 平台作为主通道，Playwright 浏览器自动化作为备用通道和 Cookie 管理工具。多 Agent 体系（ClassifyAgent / PriceAgent / ProductAgent / DefaultAgent）通过三级意图路由分发消息。SQLite 存储对话历史和商品缓存。

**Tech Stack:** Python 3.11+, websockets, playwright, httpx, openai SDK, SQLite, loguru, PyYAML

---

## Task 1: 项目初始化与依赖配置

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `config/settings.yaml`
- Create: `config/products.yaml`

**Step 1: 创建 requirements.txt**

```
websockets>=12.0
playwright>=1.40.0
httpx>=0.25.0
openai>=1.12.0
loguru>=0.7.0
PyYAML>=6.0
python-dotenv>=1.0.0
```

**Step 2: 创建 .env.example**

```bash
# 必需
API_KEY=your_qwen_api_key
COOKIES_STR=your_xianyu_cookies

# 可选
MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-max
TOGGLE_KEYWORDS=#manual
SIMULATE_HUMAN_TYPING=False
LOG_LEVEL=DEBUG
HEARTBEAT_INTERVAL=15
HEARTBEAT_TIMEOUT=5
TOKEN_REFRESH_INTERVAL=3600
TOKEN_RETRY_INTERVAL=300
MANUAL_MODE_TIMEOUT=3600
MESSAGE_EXPIRE_TIME=300000
```

**Step 3: 创建 config/settings.yaml**

```yaml
llm:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-max"

reply:
  max_length: 100
  simulate_typing: true
  typing_speed: 5

logging:
  level: "DEBUG"
  rotation: "1 day"
  retention: "30 days"
```

**Step 4: 创建 config/products.yaml**

```yaml
products: []
# 示例:
# - item_id: "123456789"
#   min_price: 4000
#   selling_points: "95新，无磕碰"
#   keywords: ["iphone", "苹果15"]
#   notes: "不包邮偏远地区"
```

**Step 5: 创建目录结构**

Run: `mkdir -p core services auth agents config/prompts storage data/logs utils`

**Step 6: Commit**

```bash
git add -A
git commit -m "chore: init project structure and dependencies"
```

---

## Task 2: 工具函数 — services/xianyu_utils.py

**Files:**
- Create: `services/__init__.py`
- Create: `services/xianyu_utils.py`
- Create: `tests/test_xianyu_utils.py`

**Step 1: 创建 services/__init__.py**

空文件。

**Step 2: 编写测试 tests/test_xianyu_utils.py**

```python
import re
import time
from services.xianyu_utils import generate_mid, generate_device_id, generate_sign, decrypt_message


def test_generate_mid_format():
    mid = generate_mid()
    # 格式: "<random><timestamp> 0"
    assert mid.endswith(" 0")
    parts = mid.split(" ")
    assert len(parts) == 2
    assert parts[0].isdigit()


def test_generate_mid_uniqueness():
    mids = {generate_mid() for _ in range(100)}
    assert len(mids) == 100


def test_generate_device_id_format():
    did = generate_device_id("12345")
    # UUID(8-4-4-4-12) + "-" + user_id
    assert did.endswith("-12345")
    uuid_part = did.replace("-12345", "")
    assert re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', uuid_part)


def test_generate_sign():
    sign = generate_sign("1710000000000", "test_token", '{"key":"value"}')
    # 应返回 32 位 hex 字符串（MD5）
    assert len(sign) == 32
    assert all(c in '0123456789abcdef' for c in sign)


def test_generate_sign_deterministic():
    s1 = generate_sign("123", "tok", "data")
    s2 = generate_sign("123", "tok", "data")
    assert s1 == s2


def test_decrypt_message_base64_json():
    import base64, json
    payload = json.dumps({"hello": "world"})
    encoded = base64.b64encode(payload.encode()).decode()
    result = decrypt_message(encoded)
    assert result["hello"] == "world"
```

**Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_xianyu_utils.py -v`
Expected: FAIL — module not found

**Step 4: 实现 services/xianyu_utils.py**

```python
"""闲鱼平台工具函数：签名、设备ID、消息解码等。"""

import base64
import hashlib
import json
import random
import re
import struct
import time
import uuid


def generate_mid() -> str:
    """生成消息 ID。格式: <random><timestamp_ms> 0"""
    random_part = int(1000 * random.random())
    timestamp = int(time.time() * 1000)
    return f"{random_part}{timestamp} 0"


def generate_device_id(user_id: str) -> str:
    """生成设备 ID。格式: UUID-userId"""
    uid = str(uuid.uuid4())
    return f"{uid}-{user_id}"


def generate_sign(t: str, token: str, data: str) -> str:
    """生成 API 请求签名。MD5(token&t&appKey&data)"""
    app_key = "34839810"
    msg = f"{token}&{t}&{app_key}&{data}"
    return hashlib.md5(msg.encode('utf-8')).hexdigest()


def _msgpack_decode(data: bytes, offset: int = 0):
    """简易 MessagePack 解码器。"""
    if offset >= len(data):
        return None, offset

    b = data[offset]

    # Positive fixint (0x00 - 0x7f)
    if b <= 0x7f:
        return b, offset + 1

    # Fixmap (0x80 - 0x8f)
    if 0x80 <= b <= 0x8f:
        count = b & 0x0f
        offset += 1
        result = {}
        for _ in range(count):
            key, offset = _msgpack_decode(data, offset)
            val, offset = _msgpack_decode(data, offset)
            result[str(key) if not isinstance(key, str) else key] = val
        return result, offset

    # Fixarray (0x90 - 0x9f)
    if 0x90 <= b <= 0x9f:
        count = b & 0x0f
        offset += 1
        result = []
        for _ in range(count):
            val, offset = _msgpack_decode(data, offset)
            result.append(val)
        return result, offset

    # Fixstr (0xa0 - 0xbf)
    if 0xa0 <= b <= 0xbf:
        length = b & 0x1f
        offset += 1
        s = data[offset:offset + length]
        return s.decode('utf-8', errors='replace'), offset + length

    # nil
    if b == 0xc0:
        return None, offset + 1

    # false
    if b == 0xc2:
        return False, offset + 1

    # true
    if b == 0xc3:
        return True, offset + 1

    # bin8
    if b == 0xc4:
        length = data[offset + 1]
        offset += 2
        return data[offset:offset + length], offset + length

    # bin16
    if b == 0xc5:
        length = struct.unpack('>H', data[offset + 1:offset + 3])[0]
        offset += 3
        return data[offset:offset + length], offset + length

    # bin32
    if b == 0xc6:
        length = struct.unpack('>I', data[offset + 1:offset + 5])[0]
        offset += 5
        return data[offset:offset + length], offset + length

    # float32
    if b == 0xca:
        val = struct.unpack('>f', data[offset + 1:offset + 5])[0]
        return val, offset + 5

    # float64
    if b == 0xcb:
        val = struct.unpack('>d', data[offset + 1:offset + 9])[0]
        return val, offset + 9

    # uint8
    if b == 0xcc:
        return data[offset + 1], offset + 2

    # uint16
    if b == 0xcd:
        val = struct.unpack('>H', data[offset + 1:offset + 3])[0]
        return val, offset + 3

    # uint32
    if b == 0xce:
        val = struct.unpack('>I', data[offset + 1:offset + 5])[0]
        return val, offset + 5

    # uint64
    if b == 0xcf:
        val = struct.unpack('>Q', data[offset + 1:offset + 9])[0]
        return val, offset + 9

    # int8
    if b == 0xd0:
        val = struct.unpack('>b', data[offset + 1:offset + 2])[0]
        return val, offset + 2

    # int16
    if b == 0xd1:
        val = struct.unpack('>h', data[offset + 1:offset + 3])[0]
        return val, offset + 3

    # int32
    if b == 0xd2:
        val = struct.unpack('>i', data[offset + 1:offset + 5])[0]
        return val, offset + 5

    # int64
    if b == 0xd3:
        val = struct.unpack('>q', data[offset + 1:offset + 9])[0]
        return val, offset + 9

    # str8
    if b == 0xd9:
        length = data[offset + 1]
        offset += 2
        return data[offset:offset + length].decode('utf-8', errors='replace'), offset + length

    # str16
    if b == 0xda:
        length = struct.unpack('>H', data[offset + 1:offset + 3])[0]
        offset += 3
        return data[offset:offset + length].decode('utf-8', errors='replace'), offset + length

    # str32
    if b == 0xdb:
        length = struct.unpack('>I', data[offset + 1:offset + 5])[0]
        offset += 5
        return data[offset:offset + length].decode('utf-8', errors='replace'), offset + length

    # array16
    if b == 0xdc:
        count = struct.unpack('>H', data[offset + 1:offset + 3])[0]
        offset += 3
        result = []
        for _ in range(count):
            val, offset = _msgpack_decode(data, offset)
            result.append(val)
        return result, offset

    # array32
    if b == 0xdd:
        count = struct.unpack('>I', data[offset + 1:offset + 5])[0]
        offset += 5
        result = []
        for _ in range(count):
            val, offset = _msgpack_decode(data, offset)
            result.append(val)
        return result, offset

    # map16
    if b == 0xde:
        count = struct.unpack('>H', data[offset + 1:offset + 3])[0]
        offset += 3
        result = {}
        for _ in range(count):
            key, offset = _msgpack_decode(data, offset)
            val, offset = _msgpack_decode(data, offset)
            result[str(key) if not isinstance(key, str) else key] = val
        return result, offset

    # map32
    if b == 0xdf:
        count = struct.unpack('>I', data[offset + 1:offset + 5])[0]
        offset += 5
        result = {}
        for _ in range(count):
            key, offset = _msgpack_decode(data, offset)
            val, offset = _msgpack_decode(data, offset)
            result[str(key) if not isinstance(key, str) else key] = val
        return result, offset

    # Negative fixint (0xe0 - 0xff)
    if b >= 0xe0:
        return b - 256, offset + 1

    # 未知类型，跳过
    return None, offset + 1


def decrypt_message(data: str) -> dict | list | None:
    """解码消息数据。先尝试 Base64+JSON，失败则 Base64+MessagePack。"""
    # 清理 base64 字符串
    cleaned = re.sub(r'[^A-Za-z0-9+/=]', '', data)
    padding = 4 - len(cleaned) % 4
    if padding != 4:
        cleaned += '=' * padding

    try:
        decoded = base64.b64decode(cleaned)
    except Exception:
        return {"error": "base64_decode_failed"}

    # 先尝试 JSON
    try:
        return json.loads(decoded.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    # 再尝试 MessagePack
    try:
        result, _ = _msgpack_decode(decoded)
        return result
    except Exception:
        pass

    # 兜底: 返回 hex
    try:
        return {"raw_utf8": decoded.decode('utf-8', errors='replace')}
    except Exception:
        return {"raw_hex": decoded.hex()}
```

**Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_xianyu_utils.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add xianyu utils (sign, device_id, msgpack decoder)"
```

---

## Task 3: 对话历史管理 — storage/context_manager.py

**Files:**
- Create: `storage/__init__.py`
- Create: `storage/context_manager.py`
- Create: `tests/test_context_manager.py`

**Step 1: 编写测试 tests/test_context_manager.py**

```python
import os
import tempfile
import pytest
from storage.context_manager import ContextManager


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    cm = ContextManager(path)
    yield cm
    cm.close()
    os.unlink(path)


def test_add_and_get_messages(db):
    db.add_message("chat1", "user1", "item1", "user", "hello")
    db.add_message("chat1", "user1", "item1", "assistant", "hi there")
    msgs = db.get_context("chat1")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["content"] == "hi there"


def test_get_context_empty(db):
    msgs = db.get_context("nonexistent")
    assert msgs == []


def test_bargain_count(db):
    assert db.get_bargain_count("chat1") == 0
    db.increment_bargain_count("chat1")
    assert db.get_bargain_count("chat1") == 1
    db.increment_bargain_count("chat1")
    assert db.get_bargain_count("chat1") == 2


def test_item_cache(db):
    db.save_item("item1", {"title": "iPhone", "price": 4500}, 4500, "test desc")
    item = db.get_item("item1")
    assert item["title"] == "iPhone"
    assert db.get_item("nonexistent") is None


def test_message_auto_cleanup(db):
    db.max_history = 5
    for i in range(10):
        db.add_message("chat1", "u1", "i1", "user", f"msg{i}")
    msgs = db.get_context("chat1")
    assert len(msgs) == 5
    assert msgs[0]["content"] == "msg5"
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_context_manager.py -v`
Expected: FAIL

**Step 3: 实现 storage/context_manager.py**

```python
"""对话历史与商品缓存管理（SQLite）。"""

import json
import sqlite3
from datetime import datetime


class ContextManager:
    def __init__(self, db_path: str = "data/chat_history.db", max_history: int = 100):
        self.db_path = db_path
        self.max_history = max_history
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_chat_id ON messages (chat_id);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON messages (timestamp);

            CREATE TABLE IF NOT EXISTS chat_bargain_counts (
                chat_id TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS items (
                item_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                price REAL,
                description TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def add_message(self, chat_id: str, user_id: str, item_id: str, role: str, content: str):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO messages (chat_id, user_id, item_id, role, content) VALUES (?, ?, ?, ?, ?)",
            (chat_id, user_id, item_id, role, content),
        )
        self.conn.commit()
        self._cleanup(chat_id)

    def _cleanup(self, chat_id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,))
        count = cur.fetchone()[0]
        if count > self.max_history:
            excess = count - self.max_history
            cur.execute(
                "DELETE FROM messages WHERE id IN ("
                "  SELECT id FROM messages WHERE chat_id = ? ORDER BY timestamp ASC LIMIT ?"
                ")",
                (chat_id, excess),
            )
            self.conn.commit()

    def get_context(self, chat_id: str) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC",
            (chat_id,),
        )
        return [{"role": row["role"], "content": row["content"]} for row in cur.fetchall()]

    def get_bargain_count(self, chat_id: str) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT count FROM chat_bargain_counts WHERE chat_id = ?", (chat_id,))
        row = cur.fetchone()
        return row["count"] if row else 0

    def increment_bargain_count(self, chat_id: str):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO chat_bargain_counts (chat_id, count, last_updated) VALUES (?, 1, CURRENT_TIMESTAMP) "
            "ON CONFLICT(chat_id) DO UPDATE SET count = count + 1, last_updated = CURRENT_TIMESTAMP",
            (chat_id,),
        )
        self.conn.commit()

    def save_item(self, item_id: str, data: dict, price: float, description: str):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO items (item_id, data, price, description) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(item_id) DO UPDATE SET data = ?, price = ?, description = ?, last_updated = CURRENT_TIMESTAMP",
            (item_id, json.dumps(data, ensure_ascii=False), price, description,
             json.dumps(data, ensure_ascii=False), price, description),
        )
        self.conn.commit()

    def get_item(self, item_id: str) -> dict | None:
        cur = self.conn.cursor()
        cur.execute("SELECT data FROM items WHERE item_id = ?", (item_id,))
        row = cur.fetchone()
        return json.loads(row["data"]) if row else None

    def close(self):
        self.conn.close()
```

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_context_manager.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add context manager (SQLite chat history + item cache)"
```

---

## Task 4: 闲鱼 HTTP API 封装 — services/xianyu_api.py

**Files:**
- Create: `services/xianyu_api.py`
- Create: `tests/test_xianyu_api.py`

**Step 1: 编写测试 tests/test_xianyu_api.py**

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from services.xianyu_api import XianyuApi


@pytest.fixture
def api():
    return XianyuApi(cookies_str="test_cookie=value", device_id="test-device-123")


def test_init(api):
    assert api.cookies_str == "test_cookie=value"
    assert api.device_id == "test-device-123"


def test_parse_cookies(api):
    cookies = api._parse_cookies("a=1; b=2; c=3")
    assert cookies == {"a": "1", "b": "2", "c": "3"}


def test_build_sign_params(api):
    params = api._build_request_params("mtop.taobao.idlemessage.pc.login.token", '{"key":"val"}')
    assert params["appKey"] == "34839810"
    assert params["api"] == "mtop.taobao.idlemessage.pc.login.token"
    assert "sign" in params
    assert "t" in params


def test_build_item_description():
    item_info = {
        "title": "iPhone 15",
        "desc": "95 new",
        "quantity": 1,
        "soldPrice": 450000,
        "skuList": [
            {"price": 450000, "quantity": 1, "propertyList": [{"valueText": "128G Black"}]}
        ],
    }
    desc = XianyuApi.build_item_description(item_info)
    assert desc["title"] == "iPhone 15"
    assert "4500" in str(desc["price_range"])
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_xianyu_api.py -v`
Expected: FAIL

**Step 3: 实现 services/xianyu_api.py**

```python
"""闲鱼 HTTP API 封装：Token 获取、商品详情查询等。"""

import json
import time
import httpx
from loguru import logger
from services.xianyu_utils import generate_sign


class XianyuApi:
    TOKEN_API = "https://h5api.m.goofish.com/h5/mtop.taobao.idlemessage.pc.login.token/1.0/"
    ITEM_API = "https://h5api.m.goofish.com/h5/mtop.taobao.idle.pc.detail/1.0/"
    LOGIN_CHECK_API = "https://passport.goofish.com/newlogin/hasLogin.do"

    def __init__(self, cookies_str: str, device_id: str):
        self.cookies_str = cookies_str
        self.device_id = device_id
        self._csrf_token = ""
        self._update_csrf_token()

    def _update_csrf_token(self):
        cookies = self._parse_cookies(self.cookies_str)
        self._csrf_token = cookies.get("_m_h5_tk", "").split("_")[0]

    @staticmethod
    def _parse_cookies(cookies_str: str) -> dict:
        result = {}
        for item in cookies_str.split(";"):
            item = item.strip()
            if "=" in item:
                key, val = item.split("=", 1)
                result[key.strip()] = val.strip()
        return result

    def _build_request_params(self, api: str, data_str: str) -> dict:
        t = str(int(time.time() * 1000))
        sign = generate_sign(t, self._csrf_token, data_str)
        return {
            "jsv": "2.7.2",
            "appKey": "34839810",
            "t": t,
            "sign": sign,
            "v": "1.0",
            "type": "originaljson",
            "accountSite": "xianyu",
            "dataType": "json",
            "timeout": "20000",
            "api": api,
            "sessionOption": "AutoLoginOnly",
        }

    async def get_token(self) -> str | None:
        data_str = json.dumps({"appKey": "444e9908a51d1cb236a27862abc769c9", "deviceId": self.device_id})
        params = self._build_request_params("mtop.taobao.idlemessage.pc.login.token", data_str)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.TOKEN_API,
                    params=params,
                    data={"data": data_str},
                    headers={"Cookie": self.cookies_str},
                    timeout=20,
                )
                result = resp.json()
                ret_values = result.get("ret", [])
                if any("SUCCESS" in r for r in ret_values):
                    token = result.get("data", {}).get("accessToken")
                    logger.info("Token obtained successfully")
                    return token
                logger.warning(f"Token request failed: {ret_values}")
                return None
        except Exception as e:
            logger.error(f"Token request error: {e}")
            return None

    async def get_item_info(self, item_id: str) -> dict | None:
        data_str = json.dumps({"itemId": item_id})
        params = self._build_request_params("mtop.taobao.idle.pc.detail", data_str)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.ITEM_API,
                    params=params,
                    data={"data": data_str},
                    headers={"Cookie": self.cookies_str},
                    timeout=20,
                )
                result = resp.json()
                return result.get("data", {}).get("itemDO")
        except Exception as e:
            logger.error(f"Item info request error: {e}")
            return None

    async def check_login(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.LOGIN_CHECK_API,
                    headers={"Cookie": self.cookies_str},
                    timeout=10,
                )
                return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def build_item_description(item_info: dict) -> dict:
        def format_price(price_fen) -> float:
            return round(float(price_fen) / 100, 2)

        sku_details = []
        prices = []
        total_stock = 0

        for sku in item_info.get("skuList", []):
            price = format_price(sku.get("price", 0))
            qty = sku.get("quantity", 0)
            specs = ", ".join(p.get("valueText", "") for p in sku.get("propertyList", []))
            sku_details.append({"spec": specs, "price": price, "stock": qty})
            prices.append(price)
            total_stock += qty

        if not prices:
            price_range = format_price(item_info.get("soldPrice", 0))
        elif len(set(prices)) == 1:
            price_range = prices[0]
        else:
            price_range = f"{min(prices)} - {max(prices)}"

        return {
            "title": item_info.get("title", ""),
            "desc": item_info.get("desc", ""),
            "price_range": price_range,
            "total_stock": total_stock or item_info.get("quantity", 0),
            "sku_details": sku_details,
        }
```

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_xianyu_api.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add xianyu API client (token, item info, login check)"
```

---

## Task 5: AI Agent 体系 — agents/

**Files:**
- Create: `agents/__init__.py`
- Create: `agents/classify_agent.py`
- Create: `agents/price_agent.py`
- Create: `agents/product_agent.py`
- Create: `agents/default_agent.py`
- Create: `agents/router.py`
- Create: `tests/test_agents.py`
- Create: `config/prompts/classify_prompt.md`
- Create: `config/prompts/price_prompt.md`
- Create: `config/prompts/product_prompt.md`
- Create: `config/prompts/default_prompt.md`
- Create: `config/prompts/global_rules.md`

**Step 1: 创建 Prompt 模板文件**

`config/prompts/global_rules.md`:

```markdown
# 全局规则

你是一个闲鱼卖家的智能客服助手。

## 基本原则
- 回复简洁友好，像真人卖家一样自然交流
- 回复长度控制在 100 字以内
- 不主动提及其他平台（微信、QQ等）
- 所有交易在闲鱼平台内完成
- 遇到不确定的问题，引导买家联系卖家本人
```

`config/prompts/classify_prompt.md`:

```markdown
你是一个意图分类器。根据买家消息，判断其意图类别。

仅输出以下类别之一，不要输出其他内容：
- price: 讨价还价、问价格、砍价
- product: 询问商品详情、参数、规格、使用方法
- default: 打招呼、闲聊、其他
- no_reply: 无需回复（如表情、图片、系统消息）
```

`config/prompts/price_prompt.md`:

```markdown
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
```

`config/prompts/product_prompt.md`:

```markdown
你是闲鱼卖家的商品咨询助手。根据商品信息回答买家的问题。

## 注意
- 如实描述商品状况，不夸大
- 不确定的信息不要编造
- 引导买家查看商品详情页
```

`config/prompts/default_prompt.md`:

```markdown
你是闲鱼卖家的客服助手。友好回复买家的各类消息。

## 注意
- 热情但不过分
- 引导买家关注商品
- 遇到售后问题，告知买家稍后卖家本人会处理
```

**Step 2: 编写测试 tests/test_agents.py**

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.router import IntentRouter
from agents.classify_agent import ClassifyAgent
from agents.price_agent import PriceAgent
from agents.product_agent import ProductAgent
from agents.default_agent import DefaultAgent


class TestIntentRouter:
    def setup_method(self):
        self.router = IntentRouter()

    def test_price_keyword_match(self):
        assert self.router.keyword_match("能便宜点吗") == "price"
        assert self.router.keyword_match("最低多少钱卖") == "price"
        assert self.router.keyword_match("500元可以吗") == "price"
        assert self.router.keyword_match("能少50吗") == "price"

    def test_product_keyword_match(self):
        assert self.router.keyword_match("参数是什么") == "product"
        assert self.router.keyword_match("什么型号") == "product"
        assert self.router.keyword_match("和iPhone14比怎么样") == "product"

    def test_product_priority_over_price(self):
        # "这个规格多少钱" 同时含商品和议价关键词，商品优先
        assert self.router.keyword_match("这个规格多少钱") == "product"

    def test_no_keyword_match(self):
        assert self.router.keyword_match("你好") is None
        assert self.router.keyword_match("在吗") is None


class TestSafetyFilter:
    def test_blocks_wechat(self):
        from agents.default_agent import BaseAgent
        assert "安全提醒" in BaseAgent.safe_filter("加我微信吧")

    def test_blocks_qq(self):
        from agents.default_agent import BaseAgent
        assert "安全提醒" in BaseAgent.safe_filter("QQ联系")

    def test_passes_normal(self):
        from agents.default_agent import BaseAgent
        assert BaseAgent.safe_filter("商品不错") == "商品不错"


class TestPriceAgentTemperature:
    def test_dynamic_temperature(self):
        agent = PriceAgent()
        assert agent.get_temperature(0) == 0.3
        assert agent.get_temperature(1) == 0.45
        assert agent.get_temperature(2) == 0.6
        assert agent.get_temperature(4) == 0.9
        assert agent.get_temperature(10) == 0.9  # capped
```

**Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_agents.py -v`
Expected: FAIL

**Step 4: 实现 Agent 基类和安全过滤 — agents/default_agent.py**

```python
"""默认回复 Agent + BaseAgent 基类。"""

import os
from openai import AsyncOpenAI
from loguru import logger


class BaseAgent:
    BLOCKED_PHRASES = ["微信", "QQ", "支付宝", "银行卡", "线下"]

    def __init__(self, prompt_path: str, temperature: float = 0.7):
        self.prompt_path = prompt_path
        self.temperature = temperature
        self.system_prompt = self._load_prompt()
        self.client = AsyncOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.model = os.getenv("MODEL_NAME", "qwen-max")

    def _load_prompt(self) -> str:
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {self.prompt_path}")
            return ""

    @staticmethod
    def safe_filter(text: str) -> str:
        for phrase in BaseAgent.BLOCKED_PHRASES:
            if phrase in text:
                return "[安全提醒] 请通过闲鱼平台沟通，不要在站外交易哦"
        return text

    async def generate(self, user_msg: str, item_desc: str = "", context: str = "", **kwargs) -> str:
        prompt = self.system_prompt
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        messages = [
            {
                "role": "system",
                "content": f"【商品信息】{item_desc}\n【对话历史】{context}\n{prompt}",
            },
            {"role": "user", "content": user_msg},
        ]

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=500,
                top_p=0.8,
            )
            reply = resp.choices[0].message.content.strip()
            return self.safe_filter(reply)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "抱歉，系统繁忙，请稍后再试"


class DefaultAgent(BaseAgent):
    def __init__(self):
        super().__init__("config/prompts/default_prompt.md", temperature=0.7)
```

**Step 5: 实现 ClassifyAgent — agents/classify_agent.py**

```python
"""意图分类 Agent。"""

from agents.default_agent import BaseAgent


class ClassifyAgent(BaseAgent):
    VALID_INTENTS = {"price", "product", "default", "no_reply"}

    def __init__(self):
        super().__init__("config/prompts/classify_prompt.md", temperature=0.3)

    async def classify(self, user_msg: str, item_desc: str = "", context: str = "") -> str:
        result = await self.generate(user_msg, item_desc, context)
        result = result.strip().lower()
        return result if result in self.VALID_INTENTS else "default"
```

**Step 6: 实现 PriceAgent — agents/price_agent.py**

```python
"""议价 Agent。动态 temperature 随议价次数递增。"""

from agents.default_agent import BaseAgent


class PriceAgent(BaseAgent):
    def __init__(self):
        super().__init__("config/prompts/price_prompt.md", temperature=0.3)

    @staticmethod
    def get_temperature(bargain_count: int) -> float:
        return min(0.3 + bargain_count * 0.15, 0.9)

    async def generate(self, user_msg: str, item_desc: str = "", context: str = "", **kwargs) -> str:
        bargain_count = kwargs.get("bargain_count", 0)
        self.temperature = self.get_temperature(bargain_count)
        return await super().generate(user_msg, item_desc, context, **kwargs)
```

**Step 7: 实现 ProductAgent — agents/product_agent.py**

```python
"""商品咨询 Agent。"""

from agents.default_agent import BaseAgent


class ProductAgent(BaseAgent):
    def __init__(self):
        super().__init__("config/prompts/product_prompt.md", temperature=0.4)
```

**Step 8: 实现意图路由 — agents/router.py**

```python
"""三级意图路由：关键词匹配 -> LLM 分类 -> Agent 分发。"""

import re
from loguru import logger
from agents.classify_agent import ClassifyAgent
from agents.price_agent import PriceAgent
from agents.product_agent import ProductAgent
from agents.default_agent import DefaultAgent


class IntentRouter:
    PRODUCT_KEYWORDS = ["参数", "规格", "型号", "连接", "对比"]
    PRODUCT_PATTERNS = [re.compile(r"和.+比")]

    PRICE_KEYWORDS = ["便宜", "价", "砍价", "少点", "多少钱", "最低"]
    PRICE_PATTERNS = [re.compile(r"\d+元"), re.compile(r"能少\d+")]

    def __init__(self):
        self.classify_agent = ClassifyAgent()
        self.agents = {
            "price": PriceAgent(),
            "product": ProductAgent(),
            "default": DefaultAgent(),
        }

    def keyword_match(self, text: str) -> str | None:
        # 商品关键词优先
        for kw in self.PRODUCT_KEYWORDS:
            if kw in text:
                return "product"
        for pat in self.PRODUCT_PATTERNS:
            if pat.search(text):
                return "product"

        # 议价关键词
        for kw in self.PRICE_KEYWORDS:
            if kw in text:
                return "price"
        for pat in self.PRICE_PATTERNS:
            if pat.search(text):
                return "price"

        return None

    async def route(self, user_msg: str, item_desc: str = "", context: str = "", **kwargs) -> str:
        # 一级：关键词匹配
        intent = self.keyword_match(user_msg)
        if intent:
            logger.info(f"Keyword match: {intent}")
        else:
            # 二级：LLM 分类
            intent = await self.classify_agent.classify(user_msg, item_desc, context)
            logger.info(f"LLM classify: {intent}")

        if intent == "no_reply":
            return ""

        # 三级：Agent 生成回复
        agent = self.agents.get(intent, self.agents["default"])
        return await agent.generate(user_msg, item_desc, context, **kwargs)
```

**Step 9: 创建 agents/__init__.py**

空文件。

**Step 10: 运行测试确认通过**

Run: `python -m pytest tests/test_agents.py -v`
Expected: ALL PASS

**Step 11: Commit**

```bash
git add -A
git commit -m "feat: add AI agent system (router, classify, price, product, default)"
```

---

## Task 6: Cookie 管理 — auth/

**Files:**
- Create: `auth/__init__.py`
- Create: `auth/cookie_manager.py`
- Create: `tests/test_cookie_manager.py`

**Step 1: 编写测试 tests/test_cookie_manager.py**

```python
import os
import json
import tempfile
import pytest
from auth.cookie_manager import CookieManager


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_load_from_env(tmp_dir, monkeypatch):
    monkeypatch.setenv("COOKIES_STR", "a=1; b=2")
    cm = CookieManager(data_dir=tmp_dir)
    assert cm.get_cookies_str() == "a=1; b=2"


def test_save_and_load_json(tmp_dir):
    cm = CookieManager(data_dir=tmp_dir)
    cm.update_cookies("x=10; y=20")
    # 重新加载
    cm2 = CookieManager(data_dir=tmp_dir)
    assert "x=10" in cm2.get_cookies_str()


def test_parse_cookies():
    cm = CookieManager.__new__(CookieManager)
    result = cm._parse_cookie_str("a=1; b=2; c=3")
    assert result == {"a": "1", "b": "2", "c": "3"}
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_cookie_manager.py -v`
Expected: FAIL

**Step 3: 实现 auth/cookie_manager.py**

```python
"""Cookie 存储、加载、回写。"""

import json
import os
from pathlib import Path
from loguru import logger


class CookieManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.data_dir / "cookies.json"
        self._cookies_str = ""
        self._load()

    def _load(self):
        # 优先从 JSON 文件加载
        if self.json_path.exists():
            try:
                with open(self.json_path, "r") as f:
                    data = json.load(f)
                self._cookies_str = data.get("cookies_str", "")
                if self._cookies_str:
                    logger.info("Cookies loaded from JSON file")
                    return
            except (json.JSONDecodeError, KeyError):
                pass

        # 其次从环境变量加载
        env_cookies = os.getenv("COOKIES_STR", "")
        if env_cookies:
            self._cookies_str = env_cookies
            logger.info("Cookies loaded from environment")
            self._save_json()

    def _save_json(self):
        with open(self.json_path, "w") as f:
            json.dump({"cookies_str": self._cookies_str}, f)

    def get_cookies_str(self) -> str:
        return self._cookies_str

    def get_cookies_dict(self) -> dict:
        return self._parse_cookie_str(self._cookies_str)

    def update_cookies(self, cookies_str: str):
        self._cookies_str = cookies_str
        self._save_json()
        logger.info("Cookies updated and saved")

    @staticmethod
    def _parse_cookie_str(cookies_str: str) -> dict:
        result = {}
        for item in cookies_str.split(";"):
            item = item.strip()
            if "=" in item:
                key, val = item.split("=", 1)
                result[key.strip()] = val.strip()
        return result
```

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_cookie_manager.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add cookie manager (load, save, update)"
```

---

## Task 7: Token 管理 — auth/token_manager.py

**Files:**
- Create: `auth/token_manager.py`
- Create: `tests/test_token_manager.py`

**Step 1: 编写测试 tests/test_token_manager.py**

```python
import time
import pytest
from unittest.mock import AsyncMock, patch
from auth.token_manager import TokenManager


@pytest.fixture
def tm():
    return TokenManager(cookies_str="test=1", device_id="dev-123")


def test_init(tm):
    assert tm.current_token is None
    assert tm.device_id == "dev-123"


def test_needs_refresh_initially(tm):
    assert tm.needs_refresh() is True


def test_needs_refresh_after_set(tm):
    tm.current_token = "tok123"
    tm.last_refresh_time = time.time()
    assert tm.needs_refresh() is False


def test_needs_refresh_after_timeout(tm):
    tm.current_token = "tok123"
    tm.last_refresh_time = time.time() - 4000  # > 3600
    assert tm.needs_refresh() is True
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_token_manager.py -v`
Expected: FAIL

**Step 3: 实现 auth/token_manager.py**

```python
"""WebSocket Token 获取与刷新管理。"""

import os
import time
from loguru import logger
from services.xianyu_api import XianyuApi


class TokenManager:
    def __init__(self, cookies_str: str, device_id: str):
        self.cookies_str = cookies_str
        self.device_id = device_id
        self.current_token: str | None = None
        self.last_refresh_time: float = 0
        self.refresh_interval = int(os.getenv("TOKEN_REFRESH_INTERVAL", "3600"))
        self.retry_interval = int(os.getenv("TOKEN_RETRY_INTERVAL", "300"))
        self.api = XianyuApi(cookies_str, device_id)

    def needs_refresh(self) -> bool:
        if not self.current_token:
            return True
        return (time.time() - self.last_refresh_time) >= self.refresh_interval

    async def refresh(self) -> bool:
        token = await self.api.get_token()
        if token:
            self.current_token = token
            self.last_refresh_time = time.time()
            logger.info("Token refreshed successfully")
            return True
        logger.warning("Token refresh failed, trying login check...")
        if await self.api.check_login():
            token = await self.api.get_token()
            if token:
                self.current_token = token
                self.last_refresh_time = time.time()
                logger.info("Token refreshed after login check")
                return True
        logger.error("Token refresh failed completely")
        return False

    def update_cookies(self, cookies_str: str):
        self.cookies_str = cookies_str
        self.api = XianyuApi(cookies_str, self.device_id)
```

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_token_manager.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add token manager (refresh, retry, login check fallback)"
```

---

## Task 8: 消息通道抽象 — core/channel.py

**Files:**
- Create: `core/__init__.py`
- Create: `core/channel.py`

**Step 1: 实现 core/channel.py**

```python
"""消息通道抽象基类。"""

from abc import ABC, abstractmethod
from typing import Callable, Awaitable


class MessageChannel(ABC):
    """消息通道基类。WebSocket 和 Playwright 通道都实现此接口。"""

    @abstractmethod
    async def connect(self):
        """建立连接。"""
        ...

    @abstractmethod
    async def disconnect(self):
        """断开连接。"""
        ...

    @abstractmethod
    async def send_message(self, chat_id: str, content: str, receiver_id: str):
        """发送消息。"""
        ...

    @abstractmethod
    async def listen(self, on_message: Callable[..., Awaitable]):
        """监听消息，收到后调用 on_message 回调。"""
        ...

    @abstractmethod
    async def is_connected(self) -> bool:
        """检查连接状态。"""
        ...
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat: add message channel abstract base class"
```

---

## Task 9: WebSocket 通道 — core/websocket_channel.py

**Files:**
- Create: `core/websocket_channel.py`
- Create: `tests/test_websocket_channel.py`

**Step 1: 编写测试 tests/test_websocket_channel.py**

```python
import json
import pytest
from core.websocket_channel import WebSocketChannel


def test_build_register_message():
    ch = WebSocketChannel.__new__(WebSocketChannel)
    ch.token = "test_token"
    ch.device_id = "dev-123"
    msg = ch._build_register_message()
    assert msg["lwp"] == "/reg"
    assert msg["headers"]["token"] == "test_token"
    assert msg["headers"]["did"] == "dev-123"
    assert msg["headers"]["app-key"] == "444e9908a51d1cb236a27862abc769c9"


def test_build_ack():
    ch = WebSocketChannel.__new__(WebSocketChannel)
    ack = ch._build_ack("mid-123", "sid-456")
    assert ack["code"] == 200
    assert ack["headers"]["mid"] == "mid-123"


def test_build_heartbeat():
    ch = WebSocketChannel.__new__(WebSocketChannel)
    hb = ch._build_heartbeat()
    assert hb["lwp"] == "/!"
    assert "mid" in hb["headers"]


def test_build_send_message():
    ch = WebSocketChannel.__new__(WebSocketChannel)
    ch.my_id = "seller123"
    msg = ch._build_send_message("chat456", "hello world", "buyer789")
    assert msg["lwp"] == "/r/MessageSend/sendByReceiverScope"
    body = msg["body"]
    assert body[0]["cid"] == "chat456@goofish"
    assert "buyer789@goofish" in body[1]["actualReceivers"]


def test_is_sync_package():
    ch = WebSocketChannel.__new__(WebSocketChannel)
    valid = {"body": {"syncPushPackage": {"data": [{"data": "test"}]}}}
    invalid = {"body": {}}
    assert ch._is_sync_package(valid) is True
    assert ch._is_sync_package(invalid) is False


def test_is_chat_message():
    ch = WebSocketChannel.__new__(WebSocketChannel)
    valid = {"1": {"5": 123, "10": {"reminderContent": "hi"}}}
    invalid = {"1": {"5": 123}}
    assert ch._is_chat_message(valid) is True
    assert ch._is_chat_message(invalid) is False


def test_parse_chat_message():
    ch = WebSocketChannel.__new__(WebSocketChannel)
    msg = {
        "1": {
            "2": "chat123@goofish_extra",
            "5": 1710000000000,
            "10": {
                "reminderTitle": "Buyer",
                "senderUserId": "buyer456",
                "reminderContent": "hello",
                "reminderUrl": "https://example.com?itemId=item789",
            },
        }
    }
    parsed = ch._parse_chat_message(msg)
    assert parsed["chat_id"] == "chat123"
    assert parsed["sender_id"] == "buyer456"
    assert parsed["content"] == "hello"
    assert parsed["item_id"] == "item789"
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_websocket_channel.py -v`
Expected: FAIL

**Step 3: 实现 core/websocket_channel.py**

```python
"""WebSocket 消息通道（主力通道）。"""

import asyncio
import base64
import json
import os
import time
import re
from typing import Callable, Awaitable
from urllib.parse import urlparse, parse_qs

import websockets
from loguru import logger

from core.channel import MessageChannel
from services.xianyu_utils import generate_mid, decrypt_message


class WebSocketChannel(MessageChannel):
    WS_URL = "wss://wss-goofish.dingtalk.com/"
    APP_KEY = "444e9908a51d1cb236a27862abc769c9"

    def __init__(self, token: str, cookies_str: str, device_id: str, my_id: str):
        self.token = token
        self.cookies_str = cookies_str
        self.device_id = device_id
        self.my_id = my_id
        self.ws = None
        self._connected = False
        self._last_heartbeat_response = time.time()
        self.heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL", "15"))
        self.heartbeat_timeout = int(os.getenv("HEARTBEAT_TIMEOUT", "5"))
        self.message_expire_time = int(os.getenv("MESSAGE_EXPIRE_TIME", "300000"))

    async def connect(self):
        headers = {
            "Cookie": self.cookies_str,
            "Host": "wss-goofish.dingtalk.com",
            "Connection": "Upgrade",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Origin": "https://www.goofish.com",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        self.ws = await websockets.connect(self.WS_URL, extra_headers=headers)
        await self._register()
        self._connected = True
        self._last_heartbeat_response = time.time()
        logger.info("WebSocket connected and registered")

    async def _register(self):
        # Phase 1: Registration
        reg_msg = self._build_register_message()
        await self.ws.send(json.dumps(reg_msg))
        logger.debug("Sent registration message")

        # Phase 2: Sync status ack
        now_ms = int(time.time() * 1000)
        sync_msg = {
            "lwp": "/r/SyncStatus/ackDiff",
            "headers": {"mid": generate_mid()},
            "body": [{
                "pipeline": "sync",
                "tooLong2Tag": "PNM,1",
                "channel": "sync",
                "topic": "sync",
                "highPts": 0,
                "pts": now_ms * 1000,
                "seq": 0,
                "timestamp": now_ms,
            }],
        }
        await self.ws.send(json.dumps(sync_msg))
        logger.debug("Sent sync ack message")

    def _build_register_message(self) -> dict:
        return {
            "lwp": "/reg",
            "headers": {
                "cache-header": "app-key token ua wv",
                "app-key": self.APP_KEY,
                "token": self.token,
                "ua": "Mozilla/5.0",
                "dt": "j",
                "wv": "im:3,au:3,sy:6",
                "sync": "0,0;0;0;",
                "did": self.device_id,
                "mid": generate_mid(),
            },
        }

    def _build_ack(self, mid: str, sid: str = "") -> dict:
        return {"code": 200, "headers": {"mid": mid, "sid": sid}}

    def _build_heartbeat(self) -> dict:
        return {"lwp": "/!", "headers": {"mid": generate_mid()}}

    def _build_send_message(self, chat_id: str, content: str, receiver_id: str) -> dict:
        encoded = base64.b64encode(json.dumps({"text": content}).encode()).decode()
        return {
            "lwp": "/r/MessageSend/sendByReceiverScope",
            "headers": {"mid": generate_mid()},
            "body": [
                {
                    "uuid": generate_mid().replace(" ", "-"),
                    "cid": f"{chat_id}@goofish",
                    "conversationType": 1,
                    "content": {
                        "contentType": 101,
                        "custom": {"type": 1, "data": encoded},
                    },
                    "redPointPolicy": 0,
                    "extension": {"extJson": "{}"},
                    "ctx": {"appVersion": "1.0", "platform": "web"},
                    "mtags": {},
                    "msgReadStatusSetting": 1,
                },
                {
                    "actualReceivers": [
                        f"{receiver_id}@goofish",
                        f"{self.my_id}@goofish",
                    ]
                },
            ],
        }

    def _is_sync_package(self, data: dict) -> bool:
        try:
            return (
                "body" in data
                and "syncPushPackage" in data["body"]
                and "data" in data["body"]["syncPushPackage"]
                and len(data["body"]["syncPushPackage"]["data"]) > 0
            )
        except (TypeError, KeyError):
            return False

    def _is_chat_message(self, message: dict) -> bool:
        try:
            return (
                "1" in message
                and isinstance(message["1"], dict)
                and "10" in message["1"]
                and "reminderContent" in message["1"]["10"]
            )
        except (TypeError, KeyError):
            return False

    def _is_system_message(self, message: dict) -> bool:
        try:
            if message.get("3", {}).get("needPush") == "false":
                return True
        except (TypeError, AttributeError):
            pass
        return False

    def _parse_chat_message(self, message: dict) -> dict:
        msg_data = message["1"]
        reminder = msg_data["10"]
        raw_chat_id = str(msg_data.get("2", ""))
        chat_id = raw_chat_id.split("@")[0] if "@" in raw_chat_id else raw_chat_id

        item_id = ""
        url = reminder.get("reminderUrl", "")
        if "itemId=" in url:
            try:
                parsed = parse_qs(urlparse(url).query)
                item_id = parsed.get("itemId", [""])[0]
            except Exception:
                match = re.search(r"itemId=(\w+)", url)
                item_id = match.group(1) if match else ""

        return {
            "chat_id": chat_id,
            "create_time": msg_data.get("5", 0),
            "sender_name": reminder.get("reminderTitle", ""),
            "sender_id": reminder.get("senderUserId", ""),
            "content": reminder.get("reminderContent", ""),
            "item_id": item_id,
        }

    async def send_message(self, chat_id: str, content: str, receiver_id: str):
        if not self.ws:
            raise ConnectionError("WebSocket not connected")
        msg = self._build_send_message(chat_id, content, receiver_id)
        await self.ws.send(json.dumps(msg))
        logger.debug(f"Sent message to {chat_id}: {content[:50]}...")

    async def listen(self, on_message: Callable[..., Awaitable]):
        if not self.ws:
            raise ConnectionError("WebSocket not connected")

        async def heartbeat_loop():
            while self._connected:
                await asyncio.sleep(self.heartbeat_interval)
                if not self._connected:
                    break
                try:
                    hb = self._build_heartbeat()
                    await self.ws.send(json.dumps(hb))
                except Exception:
                    break
                # Check heartbeat timeout
                if (time.time() - self._last_heartbeat_response) > (self.heartbeat_interval + self.heartbeat_timeout):
                    logger.warning("Heartbeat timeout, connection may be lost")
                    self._connected = False
                    break

        heartbeat_task = asyncio.create_task(heartbeat_loop())

        try:
            async for raw_msg in self.ws:
                try:
                    data = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue

                # ACK all messages with mid
                mid = data.get("headers", {}).get("mid")
                sid = data.get("headers", {}).get("sid", "")
                if mid:
                    ack = self._build_ack(mid, sid)
                    await self.ws.send(json.dumps(ack))

                # Heartbeat response
                if data.get("code") == 200 and mid:
                    self._last_heartbeat_response = time.time()
                    continue

                # Check for sync package
                if not self._is_sync_package(data):
                    continue

                # Extract and decode message
                sync_data = data["body"]["syncPushPackage"]["data"][0]
                raw_data = sync_data.get("data", "")

                # Try base64+JSON first
                try:
                    decoded = base64.b64decode(raw_data).decode("utf-8")
                    message = json.loads(decoded)
                except Exception:
                    message = decrypt_message(raw_data)

                if not message or not isinstance(message, dict):
                    continue

                # Filter message types
                if self._is_system_message(message):
                    continue

                if not self._is_chat_message(message):
                    continue

                # Parse and validate
                parsed = self._parse_chat_message(message)

                # Check message expiry
                now_ms = int(time.time() * 1000)
                if parsed["create_time"] and (now_ms - parsed["create_time"]) > self.message_expire_time:
                    logger.debug(f"Skipping expired message from {parsed['sender_name']}")
                    continue

                await on_message(parsed)

        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        finally:
            self._connected = False
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    async def disconnect(self):
        self._connected = False
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def is_connected(self) -> bool:
        return self._connected and self.ws is not None
```

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_websocket_channel.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add WebSocket channel (connect, register, heartbeat, send, listen)"
```

---

## Task 10: Playwright 通道 — core/browser_channel.py + auth/cookie_refresher.py

**Files:**
- Create: `core/browser_channel.py`
- Create: `auth/cookie_refresher.py`

**Step 1: 实现 auth/cookie_refresher.py**

```python
"""Playwright Cookie 续期与登录管理。"""

import asyncio
import random
from pathlib import Path
from playwright.async_api import async_playwright, Browser, Page
from loguru import logger
from auth.cookie_manager import CookieManager


class CookieRefresher:
    LOGIN_URL = "https://www.goofish.com"
    REFRESH_PAGES = [
        "https://www.goofish.com",
        "https://www.goofish.com/im",
        "https://www.goofish.com/myfish",
    ]

    def __init__(self, cookie_manager: CookieManager):
        self.cookie_manager = cookie_manager
        self.browser: Browser | None = None
        self.page: Page | None = None

    async def init_browser(self):
        pw = await async_playwright().start()
        self.browser = await pw.chromium.launch(headless=False)
        context = await self.browser.new_context()
        # Load existing cookies if available
        cookies_dict = self.cookie_manager.get_cookies_dict()
        if cookies_dict:
            cookie_list = [
                {"name": k, "value": v, "domain": ".goofish.com", "path": "/"}
                for k, v in cookies_dict.items()
            ]
            await context.add_cookies(cookie_list)
        self.page = await context.new_page()

    async def login(self) -> str:
        """打开登录页等待用户扫码，返回 Cookie 字符串。"""
        if not self.browser:
            await self.init_browser()
        await self.page.goto(self.LOGIN_URL, wait_until="networkidle")
        logger.info("Please scan QR code to login...")

        # Wait for login (check for user avatar or specific cookie)
        await self.page.wait_for_selector('[class*="avatar"], [class*="user"]', timeout=120000)

        cookies = await self.page.context.cookies()
        cookies_str = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
        self.cookie_manager.update_cookies(cookies_str)
        logger.info("Login successful, cookies saved")
        return cookies_str

    async def refresh_cookies(self) -> str | None:
        """定期访问页面刷新 Cookie。"""
        if not self.browser:
            await self.init_browser()

        # Random delay
        await asyncio.sleep(random.uniform(1, 5))

        # Visit random page
        url = random.choice(self.REFRESH_PAGES)
        try:
            await self.page.goto(url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(random.uniform(2, 5))

            cookies = await self.page.context.cookies()
            cookies_str = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
            self.cookie_manager.update_cookies(cookies_str)
            logger.debug(f"Cookies refreshed via {url}")
            return cookies_str
        except Exception as e:
            logger.error(f"Cookie refresh failed: {e}")
            return None

    async def refresh_loop(self, on_refresh: callable = None):
        """Cookie 续期循环。20~45 分钟随机间隔。"""
        while True:
            interval = random.uniform(20 * 60, 45 * 60)
            await asyncio.sleep(interval)
            result = await self.refresh_cookies()
            if result and on_refresh:
                await on_refresh(result)

    async def close(self):
        if self.browser:
            await self.browser.close()
            self.browser = None
```

**Step 2: 实现 core/browser_channel.py**

```python
"""Playwright 浏览器消息通道（备用通道）。"""

import asyncio
from typing import Callable, Awaitable
from playwright.async_api import Page
from loguru import logger
from core.channel import MessageChannel


class BrowserChannel(MessageChannel):
    IM_URL = "https://www.goofish.com/im"

    def __init__(self, page: Page):
        self.page = page
        self._connected = False
        self._poll_interval = 3

    async def connect(self):
        await self.page.goto(self.IM_URL, wait_until="networkidle")
        self._connected = True
        logger.info("Browser channel connected")

    async def disconnect(self):
        self._connected = False

    async def send_message(self, chat_id: str, content: str, receiver_id: str):
        """通过 DOM 操作发送消息。"""
        try:
            input_selector = 'textarea, [contenteditable="true"], input[type="text"]'
            await self.page.fill(input_selector, content)
            send_btn = await self.page.query_selector('button:has-text("Send"), button:has-text("发送")')
            if send_btn:
                await send_btn.click()
            else:
                await self.page.keyboard.press("Enter")
            logger.debug(f"Browser sent message: {content[:50]}...")
        except Exception as e:
            logger.error(f"Browser send failed: {e}")

    async def listen(self, on_message: Callable[..., Awaitable]):
        """DOM 轮询检测新消息。"""
        seen_messages = set()
        while self._connected:
            try:
                # This is a simplified implementation
                # Real implementation needs to match the actual DOM structure
                elements = await self.page.query_selector_all('[class*="message-content"]')
                for el in elements:
                    text = await el.inner_text()
                    msg_id = hash(text)
                    if msg_id not in seen_messages:
                        seen_messages.add(msg_id)
                        await on_message({"content": text, "chat_id": "", "sender_id": ""})
            except Exception as e:
                logger.debug(f"Browser poll error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def is_connected(self) -> bool:
        return self._connected
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add Playwright channel and cookie refresher"
```

---

## Task 11: 应用入口 — main.py

**Files:**
- Create: `main.py`
- Create: `utils/__init__.py`

**Step 1: 实现 main.py**

```python
"""闲鱼智能客服应用入口。"""

import asyncio
import argparse
import os
import random
import time
import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from auth.cookie_manager import CookieManager
from auth.cookie_refresher import CookieRefresher
from auth.token_manager import TokenManager
from core.websocket_channel import WebSocketChannel
from agents.router import IntentRouter
from services.xianyu_api import XianyuApi
from services.xianyu_utils import generate_device_id
from storage.context_manager import ContextManager


# Configure loguru
logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "DEBUG"))
logger.add(
    "data/logs/{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)


class GoofishCustomerService:
    def __init__(self):
        self.cookie_manager = CookieManager()
        self.context_manager = ContextManager()
        self.router = IntentRouter()

        cookies_str = self.cookie_manager.get_cookies_str()
        if not cookies_str:
            logger.error("No cookies found. Run with --login first.")
            sys.exit(1)

        # Extract user ID from cookies
        cookies_dict = self.cookie_manager.get_cookies_dict()
        self.my_id = cookies_dict.get("unb", "")
        self.device_id = generate_device_id(self.my_id)

        self.token_manager = TokenManager(cookies_str, self.device_id)
        self.api = XianyuApi(cookies_str, self.device_id)
        self.cookie_refresher = CookieRefresher(self.cookie_manager)

        # Manual mode state
        self.manual_mode: dict[str, float] = {}
        self.toggle_keywords = os.getenv("TOGGLE_KEYWORDS", "#manual").split(",")
        self.manual_timeout = int(os.getenv("MANUAL_MODE_TIMEOUT", "3600"))

        # Typing simulation
        self.simulate_typing = os.getenv("SIMULATE_HUMAN_TYPING", "False").lower() == "true"

        # Channel
        self.channel: WebSocketChannel | None = None
        self.connection_restart_flag = False

    def _is_manual_mode(self, chat_id: str) -> bool:
        if chat_id not in self.manual_mode:
            return False
        if (time.time() - self.manual_mode[chat_id]) > self.manual_timeout:
            del self.manual_mode[chat_id]
            logger.info(f"Manual mode expired for {chat_id}")
            return False
        return True

    def _toggle_manual_mode(self, chat_id: str) -> bool:
        if chat_id in self.manual_mode:
            del self.manual_mode[chat_id]
            logger.info(f"Manual mode OFF for {chat_id}")
            return False
        else:
            self.manual_mode[chat_id] = time.time()
            logger.info(f"Manual mode ON for {chat_id}")
            return True

    async def _simulate_typing_delay(self, text: str):
        if not self.simulate_typing:
            return
        base_delay = random.uniform(0, 1)
        typing_delay = len(text) * random.uniform(0.1, 0.3)
        total = min(base_delay + typing_delay, 10.0)
        await asyncio.sleep(total)

    async def _get_item_info(self, item_id: str) -> dict:
        """获取商品信息（缓存优先）。"""
        if not item_id:
            return {}
        cached = self.context_manager.get_item(item_id)
        if cached:
            return cached
        info = await self.api.get_item_info(item_id)
        if info:
            desc = XianyuApi.build_item_description(info)
            self.context_manager.save_item(item_id, desc, desc.get("price_range", 0), desc.get("desc", ""))
            return desc
        return {}

    async def _on_message(self, parsed: dict):
        chat_id = parsed["chat_id"]
        sender_id = parsed["sender_id"]
        content = parsed["content"]
        item_id = parsed["item_id"]

        # Self message (seller)
        if sender_id == self.my_id:
            # Check for manual mode toggle
            if any(kw in content for kw in self.toggle_keywords):
                self._toggle_manual_mode(chat_id)
                return
            # Record seller message as assistant
            self.context_manager.add_message(chat_id, sender_id, item_id, "assistant", content)
            return

        # Buyer message
        logger.info(f"[{parsed['sender_name']}] {content}")

        # Record buyer message
        self.context_manager.add_message(chat_id, sender_id, item_id, "user", content)

        # Check manual mode
        if self._is_manual_mode(chat_id):
            logger.debug(f"Manual mode active for {chat_id}, skipping auto-reply")
            return

        # Get item info and context
        item_info = await self._get_item_info(item_id)
        item_desc = str(item_info) if item_info else ""
        context_msgs = self.context_manager.get_context(chat_id)
        context_str = "\n".join(f"{m['role']}: {m['content']}" for m in context_msgs[-10:])

        bargain_count = self.context_manager.get_bargain_count(chat_id)

        # Route and generate reply
        reply = await self.router.route(
            content,
            item_desc=item_desc,
            context=context_str,
            bargain_count=bargain_count,
            min_price=item_info.get("min_price", ""),
            product_name=item_info.get("title", ""),
            price=item_info.get("price_range", ""),
            description=item_info.get("desc", ""),
        )

        if not reply:
            return

        # Check if price intent -> increment bargain count
        intent = self.router.keyword_match(content)
        if intent == "price":
            self.context_manager.increment_bargain_count(chat_id)

        # Simulate typing
        await self._simulate_typing_delay(reply)

        # Send reply
        await self.channel.send_message(chat_id, reply, sender_id)
        self.context_manager.add_message(chat_id, self.my_id, item_id, "assistant", reply)
        logger.info(f"[Reply] {reply}")

    async def _token_refresh_loop(self):
        while True:
            await asyncio.sleep(60)
            if self.token_manager.needs_refresh():
                success = await self.token_manager.refresh()
                if success:
                    self.connection_restart_flag = True

    async def run(self):
        """主运行循环。"""
        max_retries = 3
        retry_count = 0

        while True:
            try:
                # Ensure token
                if self.token_manager.needs_refresh():
                    success = await self.token_manager.refresh()
                    if not success:
                        logger.error("Failed to obtain token")
                        await asyncio.sleep(30)
                        continue

                # Create WebSocket channel
                self.channel = WebSocketChannel(
                    token=self.token_manager.current_token,
                    cookies_str=self.cookie_manager.get_cookies_str(),
                    device_id=self.device_id,
                    my_id=self.my_id,
                )

                await self.channel.connect()
                retry_count = 0
                self.connection_restart_flag = False

                # Start background tasks
                token_task = asyncio.create_task(self._token_refresh_loop())
                cookie_task = asyncio.create_task(
                    self.cookie_refresher.refresh_loop(
                        on_refresh=lambda c: self.token_manager.update_cookies(c)
                    )
                )

                # Listen for messages
                await self.channel.listen(self._on_message)

            except Exception as e:
                logger.error(f"Connection error: {e}")
                retry_count += 1

            finally:
                if self.channel:
                    await self.channel.disconnect()
                # Cancel background tasks
                for task in [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]:
                    task.cancel()

            if self.connection_restart_flag:
                logger.info("Restarting connection with new token...")
                self.connection_restart_flag = False
                retry_count = 0
                continue

            if retry_count >= max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded, switching to browser mode...")
                # TODO: Switch to Playwright channel
                break

            wait_time = 5
            logger.info(f"Reconnecting in {wait_time}s...")
            await asyncio.sleep(wait_time)


async def do_login():
    cm = CookieManager()
    cr = CookieRefresher(cm)
    await cr.login()
    await cr.close()


def main():
    parser = argparse.ArgumentParser(description="Goofish Customer Service")
    parser.add_argument("--login", action="store_true", help="Login and save cookies")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    args = parser.parse_args()

    if args.login:
        asyncio.run(do_login())
        return

    service = GoofishCustomerService()
    asyncio.run(service.run())


if __name__ == "__main__":
    main()
```

**Step 2: 创建 utils/__init__.py**

空文件。

**Step 3: 运行全部测试**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add main entry point with dual-channel support and manual mode"
```

---

## Task 12: 集成验证与清理

**Step 1: 确认目录结构完整**

Run: `find . -name "*.py" -o -name "*.yaml" -o -name "*.md" | grep -v __pycache__ | grep -v XianyuAutoAgent | grep -v .git | sort`

Expected: 所有计划文件都已创建。

**Step 2: 运行全部测试**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 3: 验证 import 链**

Run: `python -c "from main import GoofishCustomerService; print('Import OK')"`
Expected: Import OK

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: integration verification complete"
```

---

## 实施顺序总结

| Task | 模块 | 依赖 |
|------|------|------|
| 1 | 项目初始化 | 无 |
| 2 | services/xianyu_utils.py | 无 |
| 3 | storage/context_manager.py | 无 |
| 4 | services/xianyu_api.py | Task 2 |
| 5 | agents/* | 无 |
| 6 | auth/cookie_manager.py | 无 |
| 7 | auth/token_manager.py | Task 4 |
| 8 | core/channel.py | 无 |
| 9 | core/websocket_channel.py | Task 2, 8 |
| 10 | core/browser_channel.py + auth/cookie_refresher.py | Task 6, 8 |
| 11 | main.py | 全部 |
| 12 | 集成验证 | 全部 |

可并行的任务组：
- **并行组 A**: Task 2 + Task 3 + Task 5 + Task 6 + Task 8（无依赖）
- **并行组 B**: Task 4 + Task 7 + Task 9 + Task 10（依赖组 A）
- **串行**: Task 11 → Task 12
