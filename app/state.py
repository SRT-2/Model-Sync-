from __future__ import annotations

from typing import Optional
from app.config import Settings, get_settings
from app.providers.base import LLMProvider
from app.providers.together import TogetherProvider


class AppState:
    def __init__(self) -> None:
        self.settings: Settings = get_settings()
        self.provider: LLMProvider = TogetherProvider(self.settings)

    def reload(self) -> None:
        self.settings = get_settings()
        self.provider = TogetherProvider(self.settings)


_state: Optional[AppState] = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state


def set_provider_override(provider: LLMProvider) -> None:
    """Helper for testing: override provider with a mock."""
    state = get_state()
    state.provider = provider
