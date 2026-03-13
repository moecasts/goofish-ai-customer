"""闲鱼智能客服应用入口。"""

import asyncio
import argparse
import os
import random
import time
import sys

from dotenv import load_dotenv
from loguru import logger

from auth.cookie_manager import CookieManager
from auth.cookie_refresher import CookieRefresher
from auth.token_manager import TokenManager
from core.websocket_channel import WebSocketChannel
from agents.router import IntentRouter
from services.xianyu_api import XianyuApi
from services.xianyu_utils import generate_device_id
from storage.context_manager import ContextManager

load_dotenv()

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
