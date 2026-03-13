import pytest


@pytest.mark.unit
def test_build_register_message(websocket_channel, app_key_ws):
    msg = websocket_channel._build_register_message()
    assert msg["lwp"] == "/reg"
    assert msg["headers"]["token"] == "test_token"
    assert msg["headers"]["did"] == "test-device-123"
    assert msg["headers"]["app-key"] == app_key_ws


@pytest.mark.unit
def test_build_ack(websocket_channel):
    ack = websocket_channel._build_ack("mid-123", "sid-456")
    assert ack["code"] == 200
    assert ack["headers"]["mid"] == "mid-123"


@pytest.mark.unit
def test_build_heartbeat(websocket_channel):
    hb = websocket_channel._build_heartbeat()
    assert hb["lwp"] == "/!"
    assert "mid" in hb["headers"]


@pytest.mark.unit
def test_build_send_message(websocket_channel, chat_id, buyer_id):
    msg = websocket_channel._build_send_message(chat_id, "hello world", buyer_id)
    assert msg["lwp"] == "/r/MessageSend/sendByReceiverScope"
    body = msg["body"]
    assert body[0]["cid"] == f"{chat_id}@goofish"
    assert f"{buyer_id}@goofish" in body[1]["actualReceivers"]


@pytest.mark.unit
def test_is_sync_package(websocket_channel):
    valid = {"body": {"syncPushPackage": {"data": [{"data": "test"}]}}}
    invalid = {"body": {}}
    assert websocket_channel._is_sync_package(valid) is True
    assert websocket_channel._is_sync_package(invalid) is False


@pytest.mark.unit
def test_is_chat_message(websocket_channel):
    valid = {"1": {"5": 123, "10": {"reminderContent": "hi"}}}
    invalid = {"1": {"5": 123}}
    assert websocket_channel._is_chat_message(valid) is True
    assert websocket_channel._is_chat_message(invalid) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "msg_type,sender,content,expected_chat",
    [
        ("chat", "buyer456", "hello", "chat123"),
        ("chat", "user789", "hi there", "chat456"),
    ],
)
def test_parse_chat_message_variations(
    websocket_channel, msg_type, sender, content, expected_chat
):
    """Test chat message parsing with various inputs."""
    msg = {
        "1": {
            "2": f"{expected_chat}@goofish_extra",
            "5": 1710000000000,
            "10": {
                "reminderTitle": "Buyer",
                "senderUserId": sender,
                "reminderContent": content,
                "reminderUrl": "https://example.com?itemId=item789",
            },
        }
    }
    parsed = websocket_channel._parse_chat_message(msg)
    assert parsed["chat_id"] == expected_chat
    assert parsed["sender_id"] == sender
    assert parsed["content"] == content
    assert parsed["item_id"] == "item789"
