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
