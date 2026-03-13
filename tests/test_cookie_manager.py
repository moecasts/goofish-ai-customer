import os
import json
import tempfile
import pytest
from auth.cookie_manager import CookieManager


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_load_from_env(tmp_dir, monkeypatch):
    monkeypatch.setenv("COOKIES_STR", "a=1; b=2")
    cm = CookieManager(data_dir=tmp_dir)
    assert cm.get_cookies_str() == "a=1; b=2"


def test_save_and_load_json(tmp_dir):
    cm = CookieManager(data_dir=tmp_dir)
    cm.update_cookies("x=10; y=20")
    # 重新加载
    cm2 = CookieManager(data_dir=tmp_dir)
    assert "x=10" in cm2.get_cookies_str()


def test_parse_cookies():
    cm = CookieManager.__new__(CookieManager)
    result = cm._parse_cookie_str("a=1; b=2; c=3")
    assert result == {"a": "1", "b": "2", "c": "3"}
