"""闲鱼 HTTP API 封装：Token 获取、商品详情查询等。"""

import json
import time
import httpx
from loguru import logger
from services.xianyu_utils import generate_sign


class XianyuApi:
    TOKEN_API = (
        "https://h5api.m.goofish.com/h5/mtop.taobao.idlemessage.pc.login.token/1.0/"
    )
    ITEM_API = "https://h5api.m.goofish.com/h5/mtop.taobao.idle.pc.detail/1.0/"
    LOGIN_CHECK_API = "https://passport.goofish.com/newlogin/hasLogin.do"

    def __init__(self, cookies_str: str, device_id: str):
        self.cookies_str = cookies_str
        self.device_id = device_id
        self._csrf_token = ""
        self._update_csrf_token()

    def _update_csrf_token(self):
        cookies = self._parse_cookies(self.cookies_str)
        self._csrf_token = cookies.get("_m_h5_tk", "").split("_")[0]

    @staticmethod
    def _parse_cookies(cookies_str: str) -> dict:
        result = {}
        for item in cookies_str.split(";"):
            item = item.strip()
            if "=" in item:
                key, val = item.split("=", 1)
                result[key.strip()] = val.strip()
        return result

    def _build_request_params(self, api: str, data_str: str) -> dict:
        t = str(int(time.time() * 1000))
        sign = generate_sign(t, self._csrf_token, data_str)
        return {
            "jsv": "2.7.2",
            "appKey": "34839810",
            "t": t,
            "sign": sign,
            "v": "1.0",
            "type": "originaljson",
            "accountSite": "xianyu",
            "dataType": "json",
            "timeout": "20000",
            "api": api,
            "sessionOption": "AutoLoginOnly",
        }

    async def get_token(self) -> str | None:
        data_str = json.dumps(
            {"appKey": "444e9908a51d1cb236a27862abc769c9", "deviceId": self.device_id}
        )
        params = self._build_request_params(
            "mtop.taobao.idlemessage.pc.login.token", data_str
        )
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.TOKEN_API,
                    params=params,
                    data={"data": data_str},
                    headers={"Cookie": self.cookies_str},
                    timeout=20,
                )
                result = resp.json()
                ret_values = result.get("ret", [])
                if any("SUCCESS" in r for r in ret_values):
                    token = result.get("data", {}).get("accessToken")
                    logger.info("Token obtained successfully")
                    return token
                logger.warning(f"Token request failed: {ret_values}")
                return None
        except Exception as e:
            logger.error(f"Token request error: {e}")
            return None

    async def get_item_info(self, item_id: str) -> dict | None:
        data_str = json.dumps({"itemId": item_id})
        params = self._build_request_params("mtop.taobao.idle.pc.detail", data_str)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.ITEM_API,
                    params=params,
                    data={"data": data_str},
                    headers={"Cookie": self.cookies_str},
                    timeout=20,
                )
                result = resp.json()
                return result.get("data", {}).get("itemDO")
        except Exception as e:
            logger.error(f"Item info request error: {e}")
            return None

    async def check_login(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.LOGIN_CHECK_API,
                    headers={"Cookie": self.cookies_str},
                    timeout=10,
                )
                return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def build_item_description(item_info: dict) -> dict:
        def format_price(price_fen) -> float:
            return round(float(price_fen) / 100, 2)

        sku_details = []
        prices = []
        total_stock = 0

        for sku in item_info.get("skuList", []):
            price = format_price(sku.get("price", 0))
            qty = sku.get("quantity", 0)
            specs = ", ".join(
                p.get("valueText", "") for p in sku.get("propertyList", [])
            )
            sku_details.append({"spec": specs, "price": price, "stock": qty})
            prices.append(price)
            total_stock += qty

        if not prices:
            price_range = format_price(item_info.get("soldPrice", 0))
        elif len(set(prices)) == 1:
            price_range = prices[0]
        else:
            price_range = f"{min(prices)} - {max(prices)}"

        return {
            "title": item_info.get("title", ""),
            "desc": item_info.get("desc", ""),
            "price_range": price_range,
            "total_stock": total_stock or item_info.get("quantity", 0),
            "sku_details": sku_details,
        }
