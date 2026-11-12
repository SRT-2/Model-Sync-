from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


ValidRole = Literal["system", "user", "assistant", "tool", "function"]


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any]] = ""
    name: Optional[str] = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"system", "user", "assistant", "tool", "function"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid role '{v}'. Must be one of {allowed}")
        return v.lower()


class ChatCompletionRequest(BaseModel):
    model: str = "smart-router"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    model_config = {
        "extra": "allow"
    }


class ChatChoiceMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = ""


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatChoiceMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[ChatCompletionUsage] = None
    routed_category: Optional[str] = None
    routed_model: Optional[str] = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "smart-router"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"
    provider: str
    configured_models: Dict[str, str]
    auth_enabled: bool
