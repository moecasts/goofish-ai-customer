"""LLM 客户端封装层。"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from loguru import logger
import os
import asyncio
import random


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

        # 从环境变量读取 LLM 参数
        default_temperature = os.getenv("DEFAULT_TEMPERATURE", "0.7")
        default_max_tokens = os.getenv("DEFAULT_MAX_TOKENS", "")

        # 转换类型
        try:
            temperature = float(default_temperature) if default_temperature else 0.7
        except ValueError:
            temperature = 0.7

        # 处理 max_tokens：空字符串表示 None（不限制）
        max_tokens = None
        if default_max_tokens and default_max_tokens.strip():
            try:
                max_tokens = int(default_max_tokens)
            except ValueError:
                max_tokens = None

        # 主 LLM
        primary = self._create_llm(
            model=os.getenv("PRIMARY_MODEL"),
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("MODEL_BASE_URL"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if primary:
            chain.append(primary)

        # 备用 LLM（可选）
        fallback = self._create_llm(
            model=os.getenv("FALLBACK_MODEL"),
            api_key=os.getenv("FALLBACK_API_KEY"),
            base_url=os.getenv("FALLBACK_BASE_URL"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if fallback:
            chain.append(fallback)

        return chain

    async def invoke(
        self,
        messages: list[BaseMessage],
        allow_empty: bool = False,
    ) -> BaseMessage:
        """
        调用 LLM，支持自动重试和降级。

        Args:
            messages: 消息列表
            allow_empty: 是否允许空回复
                - False (默认): 空回复视为失败，使用兜底或重试
                - True: 允许空回复（用于状态查询等特殊场景）

        Returns:
            LLM 响应消息
        """
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        base_delay = float(os.getenv("LLM_RETRY_DELAY", "1.0"))

        # 遍历 LLM 链，返回第一个成功的
        for idx, llm in enumerate(self.llm_chain):
            for attempt in range(max_retries):
                try:
                    # 注意：不再传递 temperature 和 max_tokens
                    # 使用 LLM 实例创建时的配置
                    response = await llm.ainvoke(
                        messages,
                        timeout=30,
                    )

                    # Debug: Log diagnostic info for empty responses
                    if not response.content or len(response.content.strip()) == 0:
                        finish_reason = (
                            response.response_metadata.get("finish_reason", "unknown")
                            if hasattr(response, "response_metadata")
                            else "unknown"
                        )
                        token_usage = (
                            response.response_metadata.get("token_usage", {})
                            if hasattr(response, "response_metadata")
                            else {}
                        )
                        reasoning_tokens = token_usage.get(
                            "completion_tokens_details", {}
                        ).get("reasoning_tokens", 0)
                        logger.warning(
                            f"LLM[{idx}] Empty response: finish_reason={finish_reason}, "
                            f"reasoning_tokens={reasoning_tokens}"
                        )

                    # 检查空响应
                    if not response.content or len(response.content.strip()) == 0:
                        if allow_empty:
                            # 调用方明确允许空回复
                            logger.debug(f"LLM[{idx}] 返回空响应（允许）")
                            return response
                        else:
                            # 重试逻辑
                            if attempt < max_retries - 1:
                                delay = base_delay * (2**attempt) + random.uniform(
                                    0, 0.5
                                )
                                logger.warning(
                                    f"LLM[{idx}] 返回空响应，{delay:.1f}秒后重试 (attempt {attempt + 1}/{max_retries})"
                                )
                                await asyncio.sleep(delay)
                                continue
                            else:
                                logger.error(
                                    f"LLM[{idx}] 重试 {max_retries} 次后仍返回空响应"
                                )
                                continue

                    # 成功！
                    logger.debug(f"LLM[{idx}] 调用成功")
                    return response

                except Exception as e:
                    # 重试逻辑
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
                        logger.warning(
                            f"LLM[{idx}] 调用失败: {e}，{delay:.1f}秒后重试 (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"LLM[{idx}] 重试 {max_retries} 次后仍失败: {e}")
                        continue

        # 所有 LLM 都失败，返回兜底回复
        if not allow_empty:
            logger.error("所有 LLM 都失败，使用兜底回复")
            return self._get_fallback_response()
        else:
            # 即使允许空，所有 LLM 都失败也应该记录
            logger.error("所有 LLM 都失败（允许空模式）")
            return AIMessage(content="")

    def _format_curl_command(
        self,
        llm: ChatOpenAI,
        messages: list[BaseMessage],
    ) -> str:
        """生成等价的 curl 命令用于调试。"""
        import json
        import shlex

        # Extract API parameters from llm instance
        # LangChain stores these in the OpenAI client inside the llm object
        api_key = getattr(llm, "api_key", None)
        if not api_key and hasattr(llm, "client"):
            api_key = getattr(llm.client, "api_key", None)
        if not api_key:
            api_key = os.getenv("API_KEY", "<API_KEY>")

        base_url = getattr(llm, "base_url", None)
        if not base_url and hasattr(llm, "client"):
            base_url = getattr(llm.client, "base_url", None)
        if not base_url:
            base_url = os.getenv("MODEL_BASE_URL", "<BASE_URL>")

        model = llm.model_name if hasattr(llm, "model_name") else llm.model

        # Format messages for OpenAI API
        formatted_messages = []
        for msg in messages:
            if msg.type == "system":
                formatted_messages.append({"role": "system", "content": msg.content})
            elif msg.type == "human":
                formatted_messages.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                formatted_messages.append({"role": "assistant", "content": msg.content})

        # 从 LLM 实例获取配置的参数
        temperature = getattr(llm, "temperature", 0.7)
        max_tokens = getattr(llm, "max_tokens", None)

        payload = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
        }

        # 只有当 max_tokens 不为 None 时才添加到 payload
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Generate compact JSON and properly escape it for shell
        json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        escaped_json = shlex.quote(json_str)

        curl_cmd = f"""curl -X POST "{base_url}/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {api_key}" \\
  -d {escaped_json}"""

        return curl_cmd

    def _create_llm(
        self,
        model: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Optional[ChatOpenAI]:
        """创建 ChatOpenAI 实例。"""
        if not model or not api_key:
            return None

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _get_fallback_response(self) -> BaseMessage:
        """获取兜底回复。"""
        fallback_msg = os.getenv("FALLBACK_REPLY", "卖家暂时离开了，回来马上回复！")
        return AIMessage(content=fallback_msg)
