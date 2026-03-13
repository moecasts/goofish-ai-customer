"""LLM 客户端封装层。"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from loguru import logger
import os


class LLMError(Exception):
    """LLM 调用失败。"""
    pass


class LLMClient:
    """LLM 客户端封装层 - 统一调用接口，支持重试和降级。"""

    def __init__(self):
        # 构建 LLM 列表，按优先级排序
        self.llm_chain = self._build_llm_chain()

    def _build_llm_chain(self) -> list[ChatOpenAI]:
        """构建 LLM 调用链，按优先级排序。"""
        chain = []

        # 主 LLM
        primary = self._create_llm(
            model=os.getenv("PRIMARY_MODEL"),
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("MODEL_BASE_URL"),
        )
        if primary:
            chain.append(primary)

        # 备用 LLM（可选）
        fallback = self._create_llm(
            model=os.getenv("FALLBACK_MODEL"),
            api_key=os.getenv("FALLBACK_API_KEY"),
            base_url=os.getenv("FALLBACK_BASE_URL"),
        )
        if fallback:
            chain.append(fallback)

        return chain

    async def invoke(
        self,
        messages: list[BaseMessage],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> BaseMessage:
        """
        调用 LLM，支持自动重试和降级。

        按优先级依次尝试 LLM，返回第一个成功的响应。
        """
        # 遍历 LLM 链，返回第一个成功的
        for idx, llm in enumerate(self.llm_chain):
            try:
                response = await llm.ainvoke(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30,
                )
                logger.debug(f"LLM[{idx}] 调用成功")
                return response

            except Exception as e:
                logger.warning(f"LLM[{idx}] 调用失败: {e}")
                # 继续尝试下一个
                continue

        # 所有 LLM 都失败，返回兜底回复
        logger.error("所有 LLM 都失败，使用兜底回复")
        return self._get_fallback_response()

    def _create_llm(
        self,
        model: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> Optional[ChatOpenAI]:
        """创建 ChatOpenAI 实例。"""
        if not model or not api_key:
            return None

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.7,
            max_tokens=500,
        )

    def _get_fallback_response(self) -> BaseMessage:
        """获取兜底回复。"""
        fallback_msg = os.getenv(
            "FALLBACK_REPLY", "卖家暂时离开了，回来马上回复！"
        )
        return AIMessage(content=fallback_msg)
