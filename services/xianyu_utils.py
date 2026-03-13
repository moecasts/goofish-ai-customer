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
    random_part = random.randint(0, 999999)
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
