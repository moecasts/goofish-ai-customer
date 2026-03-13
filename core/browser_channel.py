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
            send_btn = await self.page.query_selector(
                'button:has-text("Send"), button:has-text("发送")'
            )
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
                elements = await self.page.query_selector_all(
                    '[class*="message-content"]'
                )
                for el in elements:
                    text = await el.inner_text()
                    msg_id = hash(text)
                    if msg_id not in seen_messages:
                        seen_messages.add(msg_id)
                        await on_message(
                            {"content": text, "chat_id": "", "sender_id": ""}
                        )
            except Exception as e:
                logger.debug(f"Browser poll error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def is_connected(self) -> bool:
        return self._connected
