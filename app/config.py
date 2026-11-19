from __future__ import annotations

from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    router_host: str = Field(default="0.0.0.0", alias="ROUTER_HOST")
    router_port: int = Field(default=8000, alias="ROUTER_PORT")
    router_api_key: Optional[str] = Field(default=None, alias="ROUTER_API_KEY")

    # Upstream keys
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    # Dedicated upstream key used by the live ModelSync evaluation harness/server.
    openrouter_router_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_ROUTER_API_KEY")
    openrouter_allow_fallbacks: bool = Field(default=False, alias="OPENROUTER_ALLOW_FALLBACKS")
    together_api_key: Optional[str] = Field(default=None, alias="TOGETHER_API_KEY")
    upstream_api_key: Optional[str] = Field(default=None, alias="UPSTREAM_API_KEY")

    # Base URLs
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    together_base_url: str = Field(
        default="https://api.together.xyz/v1", alias="TOGETHER_BASE_URL"
    )
    upstream_base_url: Optional[str] = Field(default=None, alias="UPSTREAM_BASE_URL")

    # Connection params
    upstream_timeout_seconds: float = Field(
        default=120.0, alias="UPSTREAM_TIMEOUT_SECONDS"
    )
    upstream_max_retries: int = Field(default=2, alias="UPSTREAM_MAX_RETRIES")

    # Routing targets
    coding_model: str = Field(
        default="deepseek/deepseek-v4.1-flash-1017", alias="CODING_MODEL"
    )
    testing_model: str = Field(
        default="deepseek/deepseek-v4-flash-0731", alias="TESTING_MODEL"
    )
    reasoning_model: str = Field(
        default="openai/gpt-5.6-luna", alias="REASONING_MODEL"
    )
    classifier_model: str = Field(
        default="z-ai/glm-5.3-flash", alias="CLASSIFIER_MODEL"
    )
    # Reasoning-capable classifier models may consume output tokens internally
    # before returning the category label. Keep this limit configurable.
    classifier_max_output_tokens: int = Field(
        default=128, alias="CLASSIFIER_MAX_OUTPUT_TOKENS"
    )

    @property
    def effective_api_key(self) -> Optional[str]:
        if self.openrouter_router_api_key:
            return self.openrouter_router_api_key
        if self.openrouter_api_key:
            return self.openrouter_api_key
        if self.together_api_key:
            return self.together_api_key
        if self.upstream_api_key:
            return self.upstream_api_key
        return None

    @property
    def effective_base_url(self) -> str:
        # Explicit OpenRouter credentials must win over a stale generic URL
        # left in older .env files; otherwise provider_name and base URL diverge.
        if self.openrouter_router_api_key or self.openrouter_api_key:
            return self.openrouter_base_url
        if self.upstream_base_url:
            return self.upstream_base_url
        if self.together_api_key:
            return self.together_base_url
        return self.openrouter_base_url

    @property
    def provider_name(self) -> str:
        if self.openrouter_router_api_key or self.openrouter_api_key:
            return "openrouter"
        if self.together_api_key:
            return "together"
        return "custom"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
