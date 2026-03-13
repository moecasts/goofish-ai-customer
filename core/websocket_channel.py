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
        self.ws = await websockets.connect(self.WS_URL, additional_headers=headers)
        await self._register()
        self._connected = True
        self._last_heartbeat_response = time.time()
        logger.info("WebSocket connected and registered")

    async def _register(self):
        # Phase 1: Registration
        reg_msg = self._build_register_message()
        await self.ws.send(json.dumps(reg_msg))
        logger.debug("Sent registration message")

        # Phase 2: Wait for registration response
        for _ in range(10):  # Read up to 10 messages looking for reg response
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=5)
                data = json.loads(raw)
                # ACK any message with mid
                mid = data.get("headers", {}).get("mid")
                sid = data.get("headers", {}).get("sid", "")
                if mid:
                    ack = self._build_ack(mid, sid)
                    await self.ws.send(json.dumps(ack))

                # Look for registration response (code 200 with lwp or reg-related)
                if data.get("code") == 200 or data.get("lwp") == "/r":
                    break
                # Some servers respond with the sync data directly
                if "body" in data and "syncPushPackage" in str(data.get("body", "")):
                    break
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for registration response")
                break
            except Exception as e:
                logger.warning(f"Error during registration: {e}")
                break

        # Phase 3: Sync status ack — use pts=0 to request all pending messages
        now_ms = int(time.time() * 1000)
        sync_msg = {
            "lwp": "/r/SyncStatus/ackDiff",
            "headers": {"mid": generate_mid()},
            "body": [
                {
                    "pipeline": "sync",
                    "tooLong2Tag": "PNM,1",
                    "channel": "sync",
                    "topic": "sync",
                    "highPts": 0,
                    "pts": 0,
                    "seq": 0,
                    "timestamp": now_ms,
                }
            ],
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
        encoded = base64.b64encode(
            json.dumps({"contentType": 1, "text": {"text": content}}).encode()
        ).decode()
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
                if (time.time() - self._last_heartbeat_response) > (
                    self.heartbeat_interval + self.heartbeat_timeout
                ):
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
                sync_items = data["body"]["syncPushPackage"]["data"]

                for idx, sync_data in enumerate(sync_items):
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
                    if (
                        parsed["create_time"]
                        and (now_ms - parsed["create_time"]) > self.message_expire_time
                    ):
                        logger.debug(
                            f"Skipping expired message from {parsed['sender_name']}"
                        )
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
