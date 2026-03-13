"""工具函数。"""

from pathlib import Path
from loguru import logger
import yaml
from typing import Any


CONFIG_PATH = Path(__file__).parent.parent / "config" / "products.yaml"


def check_safety(text: str) -> str:
    """
    安全过滤。

    检测敏感词（微信/QQ/支付宝/银行卡/线下交易），
    命中则替换为平台沟通提醒。
    """
    if not text or not isinstance(text, str):
        return text

    blocked_phrases = ["微信", "QQ", "支付宝", "银行卡", "线下"]

    text_lower = text.lower()
    for phrase in blocked_phrases:
        if phrase.lower() in text_lower:
            logger.info(f"检测到敏感词: {phrase}")
            return "[安全提醒] 请通过闲鱼平台沟通，不要在站外交易哦"

    return text


async def get_item_info(item_id: str) -> dict[str, Any]:
    """
    查询商品信息。

    从配置文件中获取商品的补充信息（最低价、卖点等）。
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for product in config.get("products", []):
            if product["item_id"] == item_id:
                return product

        logger.warning(f"未找到商品 {item_id} 的配置信息")
        return {}

    except FileNotFoundError:
        logger.error(f"{CONFIG_PATH} 文件未找到")
        return {}
    except Exception as e:
        logger.error(f"读取商品配置失败: {e}")
        return {}
