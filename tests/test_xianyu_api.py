import pytest
from services.xianyu_api import XianyuApi


@pytest.fixture
def xianyu_api(device_id):
    """Create XianyuApi instance for testing."""
    return XianyuApi(cookies_str="test_cookie=value", device_id=device_id)


@pytest.mark.unit
class TestXianyuApi:
    """Test suite for XianyuApi class."""

    def test_init(self, xianyu_api, device_id):
        """Test XianyuApi initialization."""
        assert xianyu_api.cookies_str == "test_cookie=value"
        assert xianyu_api.device_id == device_id

    def test_parse_cookies(self, xianyu_api):
        """Test cookie parsing functionality."""
        cookies = xianyu_api._parse_cookies("a=1; b=2; c=3")
        assert cookies == {"a": "1", "b": "2", "c": "3"}

    def test_build_sign_params(self, xianyu_api, app_key_api):
        """Test building signed request parameters."""
        params = xianyu_api._build_request_params(
            "mtop.taobao.idlemessage.pc.login.token", '{"key":"val"}'
        )
        assert params["appKey"] == app_key_api
        assert params["api"] == "mtop.taobao.idlemessage.pc.login.token"
        assert "sign" in params
        assert "t" in params

    @pytest.mark.parametrize(
        "item_info,expected_title,expected_price",
        [
            (
                {
                    "title": "iPhone 15",
                    "desc": "95 new",
                    "quantity": 1,
                    "soldPrice": 450000,
                    "skuList": [
                        {
                            "price": 450000,
                            "quantity": 1,
                            "propertyList": [{"valueText": "128G Black"}],
                        }
                    ],
                },
                "iPhone 15",
                "4500",
            ),
            (
                {
                    "title": "MacBook Pro",
                    "desc": "Like new",
                    "quantity": 1,
                    "soldPrice": 1200000,
                    "skuList": [
                        {
                            "price": 1200000,
                            "quantity": 1,
                            "propertyList": [{"valueText": "M3 Max"}],
                        }
                    ],
                },
                "MacBook Pro",
                "12000",
            ),
        ],
    )
    def test_build_item_description(self, item_info, expected_title, expected_price):
        """Test building item descriptions from API data."""
        desc = XianyuApi.build_item_description(item_info)
        assert desc["title"] == expected_title
        assert expected_price in str(desc["price_range"])
