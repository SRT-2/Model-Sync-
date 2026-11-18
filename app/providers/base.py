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


    async def classify_with_metadata(
        self,
        classifier_model: str,
        prompt: str,
    ) -> Dict[str, Any]:
        """Classify with optional provider usage metadata.

        Providers that do not expose upstream metadata remain compatible. A
        live provider should override this method to return its generation ID,
        token usage, cost, latency, and fallback state.
        """
        category = await self.classify(classifier_model, prompt)
        return {
            "category": category,
            "raw_output": None,
            "model": classifier_model,
            "generation_id": None,
            "usage": {},
            "latency_ms": None,
            "fallback": False,
            "metadata_available": False,
        }
