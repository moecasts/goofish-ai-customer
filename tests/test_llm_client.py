"""测试 LLMClient。"""

import os
import pytest
from dotenv import load_dotenv
from services.llm_client import LLMClient, LLMError
from langchain_core.messages import HumanMessage

load_dotenv()


@pytest.mark.unit
class TestLLMClient:
    """测试 LLMClient 类。"""

    def test_build_llm_chain_with_primary_only(self, monkeypatch):
        """测试只配置主 LLM 的情况。"""
        # 使用 monkeypatch 设置环境变量
        monkeypatch.setenv("PRIMARY_MODEL", "test-model")
        monkeypatch.setenv("API_KEY", "test-key")
        monkeypatch.setenv("MODEL_BASE_URL", "https://test.com")

        client = LLMClient()
        assert len(client.llm_chain) == 1

        # 无需清理，monkeypatch 会在测试后自动恢复

    def test_build_llm_chain_with_fallback(self, monkeypatch):
        """测试配置主 LLM 和备用 LLM 的情况。"""
        monkeypatch.setenv("PRIMARY_MODEL", "test-model")
        monkeypatch.setenv("API_KEY", "test-key")
        monkeypatch.setenv("MODEL_BASE_URL", "https://test.com")
        monkeypatch.setenv("FALLBACK_MODEL", "fallback-model")
        monkeypatch.setenv("FALLBACK_API_KEY", "fallback-key")
        monkeypatch.setenv("FALLBACK_BASE_URL", "https://fallback.com")

        client = LLMClient()
        assert len(client.llm_chain) == 2

        # 无需清理，monkeypatch 会在测试后自动恢复

    def test_create_llm_without_model(self):
        """测试没有模型配置时返回 None。"""
        client = LLMClient()
        result = client._create_llm(None, None, None)
        assert result is None

    def test_get_fallback_response(self):
        """测试获取兜底回复。"""
        client = LLMClient()
        response = client._get_fallback_response()
        assert response.content == "卖家暂时离开了，回来马上回复！"

    @pytest.mark.skipif(not os.getenv("API_KEY"), reason="API_KEY not set")
    @pytest.mark.asyncio
    async def test_invoke_with_real_api(self):
        """集成测试：使用真实 API 调用。"""
        os.environ["PRIMARY_MODEL"] = os.getenv("MODEL_NAME", "qwen-max")

        client = LLMClient()
        messages = [HumanMessage(content="你好")]

        response = await client.invoke(messages, temperature=0.7)

        assert hasattr(response, "content")
        assert len(response.content) > 0
