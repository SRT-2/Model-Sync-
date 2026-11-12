from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Perform a non-streaming chat completion."""
        pass

    @abstractmethod
    async def stream_complete(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Perform a streaming chat completion yielding chunk dictionaries."""
        pass

    @abstractmethod
    async def classify(
        self,
        classifier_model: str,
        prompt: str,
    ) -> str:
        """Classify user prompt into one of the designated categories."""
        pass
