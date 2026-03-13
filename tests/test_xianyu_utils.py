import re
import pytest
from services.xianyu_utils import (
    generate_mid,
    generate_device_id,
    generate_sign,
    decrypt_message,
)


@pytest.mark.unit
class TestGenerateMid:
    """Test suite for message ID generation."""

    def test_format(self):
        """Test MID format structure."""
        mid = generate_mid()
        # Format: "<random><timestamp> 0"
        assert mid.endswith(" 0")
        parts = mid.split(" ")
        assert len(parts) == 2
        assert parts[0].isdigit()

    def test_uniqueness(self):
        """Test that generated MIDs are unique."""
        mids = {generate_mid() for _ in range(100)}
        assert len(mids) == 100


@pytest.mark.unit
class TestGenerateDeviceId:
    """Test suite for device ID generation."""

    @pytest.mark.parametrize(
        "user_id,expected_suffix",
        [
            ("12345", "-12345"),
            ("seller123", "-seller123"),
            ("buyer789", "-buyer789"),
        ],
    )
    def test_format(self, user_id, expected_suffix):
        """Test device ID format structure."""
        did = generate_device_id(user_id)
        # UUID(8-4-4-4-12) + "-" + user_id
        assert did.endswith(expected_suffix)
        uuid_part = did.replace(expected_suffix, "")
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", uuid_part
        )


@pytest.mark.unit
class TestGenerateSign:
    """Test suite for signature generation."""

    def test_sign_format(self):
        """Test that sign is a 32-character hex string (MD5)."""
        sign = generate_sign("1710000000000", "test_token", '{"key":"value"}')
        assert len(sign) == 32
        assert all(c in "0123456789abcdef" for c in sign)

    def test_deterministic(self):
        """Test that same inputs produce same signature."""
        s1 = generate_sign("123", "tok", "data")
        s2 = generate_sign("123", "tok", "data")
        assert s1 == s2

    @pytest.mark.parametrize(
        "timestamp,token,data",
        [
            ("1710000000000", "test_token", '{"key":"value"}'),
            ("1234567890", "another_token", '{"different":"data"}'),
            ("0", "empty", "{}"),
        ],
    )
    def test_various_inputs(self, timestamp, token, data):
        """Test signature generation with various inputs."""
        sign = generate_sign(timestamp, token, data)
        assert len(sign) == 32
        assert all(c in "0123456789abcdef" for c in sign)


@pytest.mark.unit
class TestDecryptMessage:
    """Test suite for message decryption."""

    def test_decrypt_base64_json(self):
        """Test decrypting base64-encoded JSON message."""
        import base64
        import json

        payload = json.dumps({"hello": "world"})
        encoded = base64.b64encode(payload.encode()).decode()
        result = decrypt_message(encoded)
        assert result["hello"] == "world"

    @pytest.mark.parametrize(
        "payload,expected_key,expected_value",
        [
            ('{"test": "value"}', "test", "value"),
            ('{"nested": {"key": "val"}}', "nested", {"key": "val"}),
            ('{"array": [1, 2, 3]}', "array", [1, 2, 3]),
        ],
    )
    def test_decrypt_various_messages(self, payload, expected_key, expected_value):
        """Test decrypting various JSON messages."""
        import base64

        encoded = base64.b64encode(payload.encode()).decode()
        result = decrypt_message(encoded)
        assert result[expected_key] == expected_value
