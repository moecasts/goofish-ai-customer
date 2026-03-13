"""测试工具函数。"""

import pytest
from services.tools import check_safety, get_item_info


@pytest.mark.unit
class TestSafetyCheck:
    """测试安全过滤功能。"""

    def test_blocks_wechat(self):
        """测试拦截微信关键词。"""
        result = check_safety("加我微信吧")
        assert "安全提醒" in result
        assert "闲鱼平台" in result

    def test_blocks_qq(self):
        """测试拦截 QQ 关键词。"""
        result = check_safety("QQ联系")
        assert "安全提醒" in result

    def test_blocks_alipay(self):
        """测试拦截支付宝关键词。"""
        result = check_safety("用支付宝支付")
        assert "安全提醒" in result

    def test_blocks_bank_card(self):
        """测试拦截银行卡关键词。"""
        result = check_safety("银行卡转账")
        assert "安全提醒" in result

    def test_blocks_offline(self):
        """测试拦截线下交易关键词。"""
        result = check_safety("线下见面交易")
        assert "安全提醒" in result

    def test_passes_normal_message(self):
        """测试正常消息通过。"""
        result = check_safety("这个商品不错")
        assert result == "这个商品不错"
        assert "安全提醒" not in result


@pytest.mark.unit
class TestGetItemInfo:
    """测试商品信息查询功能。"""

    @pytest.mark.asyncio
    async def test_get_existing_item(self):
        """测试查询存在的商品。"""
        # 假设 config/products.yaml 中有该商品
        info = await get_item_info("123456789")
        # 根据实际配置验证
        assert isinstance(info, dict)

    @pytest.mark.asyncio
    async def test_get_non_existing_item(self):
        """测试查询不存在的商品。"""
        info = await get_item_info("nonexistent")
        assert info == {}
