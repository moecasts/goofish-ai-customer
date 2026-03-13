import pytest
from services.xianyu_api import XianyuApi


@pytest.fixture
def api():
    return XianyuApi(cookies_str="test_cookie=value", device_id="test-device-123")


def test_init(api):
    assert api.cookies_str == "test_cookie=value"
    assert api.device_id == "test-device-123"


def test_parse_cookies(api):
    cookies = api._parse_cookies("a=1; b=2; c=3")
    assert cookies == {"a": "1", "b": "2", "c": "3"}


def test_build_sign_params(api):
    params = api._build_request_params(
        "mtop.taobao.idlemessage.pc.login.token", '{"key":"val"}'
    )
    assert params["appKey"] == "34839810"
    assert params["api"] == "mtop.taobao.idlemessage.pc.login.token"
    assert "sign" in params
    assert "t" in params


def test_build_item_description():
    item_info = {
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
    }
    desc = XianyuApi.build_item_description(item_info)
    assert desc["title"] == "iPhone 15"
    assert "4500" in str(desc["price_range"])
