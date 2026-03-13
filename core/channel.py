"""消息通道抽象基类。"""

from abc import ABC, abstractmethod
from typing import Callable, Awaitable


class MessageChannel(ABC):
    """消息通道基类。WebSocket 和 Playwright 通道都实现此接口。"""

    @abstractmethod
    async def connect(self):
        """建立连接。"""
        ...

    @abstractmethod
    async def disconnect(self):
        """断开连接。"""
        ...

    @abstractmethod
    async def send_message(self, chat_id: str, content: str, receiver_id: str):
        """发送消息。"""
        ...

    @abstractmethod
    async def listen(self, on_message: Callable[..., Awaitable]):
        """监听消息，收到后调用 on_message 回调。"""
        ...

    @abstractmethod
    async def is_connected(self) -> bool:
        """检查连接状态。"""
        ...
