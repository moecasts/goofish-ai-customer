"""Cookie 存储、加载、回写。"""

import json
import os
from pathlib import Path
from loguru import logger


class CookieManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.data_dir / "cookies.json"
        self._cookies_str = ""
        self._load()

    def _load(self):
        # 优先从 JSON 文件加载
        if self.json_path.exists():
            try:
                with open(self.json_path, "r") as f:
                    data = json.load(f)
                self._cookies_str = data.get("cookies_str", "")
                if self._cookies_str:
                    logger.info("Cookies loaded from JSON file")
                    return
            except (json.JSONDecodeError, KeyError):
                pass

        # 其次从环境变量加载
        env_cookies = os.getenv("COOKIES_STR", "")
        if env_cookies:
            self._cookies_str = env_cookies
            logger.info("Cookies loaded from environment")
            self._save_json()

    def _save_json(self):
        with open(self.json_path, "w") as f:
            json.dump({"cookies_str": self._cookies_str}, f)

    def get_cookies_str(self) -> str:
        return self._cookies_str

    def get_cookies_dict(self) -> dict:
        return self._parse_cookie_str(self._cookies_str)

    def update_cookies(self, cookies_str: str):
        self._cookies_str = cookies_str
        self._save_json()
        logger.info("Cookies updated and saved")

    @staticmethod
    def _parse_cookie_str(cookies_str: str) -> dict:
        result = {}
        for item in cookies_str.split(";"):
            item = item.strip()
            if "=" in item:
                key, val = item.split("=", 1)
                result[key.strip()] = val.strip()
        return result
