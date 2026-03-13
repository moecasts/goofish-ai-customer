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
