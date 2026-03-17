"""Agent 状态定义。"""

from typing import Annotated, Any, Sequence, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ItemInfo(BaseModel):
    """商品信息。"""

    item_id: str = Field(description="商品ID")
    title: str = Field(description="商品标题", min_length=1)
    price: float = Field(description="商品价格", gt=0)
    min_price: Optional[float] = Field(None, description="最低价", ge=0)
    description: Optional[str] = Field(None, description="商品描述")


class AgentState(TypedDict):
    """Agent 状态定义。"""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    intent: str
    bargain_count: int
    item_info: Optional[dict[str, Any]]
    manual_mode: bool


__all__ = ["AgentState", "ItemInfo"]
