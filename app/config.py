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
        default="qwen/qwen3.8-27b:free", alias="CODING_MODEL"
    )
    testing_model: str = Field(
        default="google/gemma-4-26b-a4b-it:free", alias="TESTING_MODEL"
    )
    reasoning_model: str = Field(
        default="nvidia/nemotron-3-ultra-550b-a55b:free", alias="REASONING_MODEL"
    )
    classifier_model: str = Field(
        default="poolside/laguna-s-2.1:free", alias="CLASSIFIER_MODEL"
    )

    @property
    def effective_api_key(self) -> Optional[str]:
        if self.openrouter_api_key:
            return self.openrouter_api_key
        if self.together_api_key:
            return self.together_api_key
        if self.upstream_api_key:
            return self.upstream_api_key
        return None

    @property
    def effective_base_url(self) -> str:
        if self.upstream_base_url:
            return self.upstream_base_url
        if self.together_api_key and not self.openrouter_api_key:
            return self.together_base_url
        return self.openrouter_base_url

    @property
    def provider_name(self) -> str:
        if self.openrouter_api_key:
            return "openrouter"
        if self.together_api_key:
            return "together"
        return "custom"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
