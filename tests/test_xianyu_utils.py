import re
from services.xianyu_utils import (
    generate_mid,
    generate_device_id,
    generate_sign,
    decrypt_message,
)


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
    assert re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", uuid_part
    )


def test_generate_sign():
    sign = generate_sign("1710000000000", "test_token", '{"key":"value"}')
    # 应返回 32 位 hex 字符串（MD5）
    assert len(sign) == 32
    assert all(c in "0123456789abcdef" for c in sign)


def test_generate_sign_deterministic():
    s1 = generate_sign("123", "tok", "data")
    s2 = generate_sign("123", "tok", "data")
    assert s1 == s2


def test_decrypt_message_base64_json():
    import base64
    import json

    payload = json.dumps({"hello": "world"})
    encoded = base64.b64encode(payload.encode()).decode()
    result = decrypt_message(encoded)
    assert result["hello"] == "world"
