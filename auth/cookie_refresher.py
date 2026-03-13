"""Playwright Cookie 续期与登录管理。"""

import asyncio
import random
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

        # Wait for login: the personal link only appears after successful login
        await self.page.wait_for_selector(
            'a[href="https://www.goofish.com/personal"]', timeout=120000
        )

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
