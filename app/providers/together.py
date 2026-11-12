from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional
from fastapi import HTTPException
from openai import AsyncOpenAI, AuthenticationError, APITimeoutError, APIConnectionError, APIError

from app.config import Settings
from app.providers.base import LLMProvider
from app.routing import CLASSIFIER_SYSTEM_PROMPT, parse_label

logger = logging.getLogger(__name__)


class TogetherProvider(LLMProvider):
    """Provider connecting to Together AI, OpenRouter, or any OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        api_key = settings.effective_api_key or "dummy-key-for-offline"
        base_url = settings.effective_base_url
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.upstream_timeout_seconds,
            max_retries=settings.upstream_max_retries,
        )

    async def classify(self, classifier_model: str, prompt: str) -> str:
        """Call classifier model to categorize the prompt."""
        messages = [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=classifier_model,
                messages=messages,
                temperature=0.0,
                max_tokens=20,
            )
            raw = response.choices[0].message.content or ""
            return parse_label(raw)
        except AuthenticationError as exc:
            logger.error("Upstream authentication failed: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"bad_upstream_api_key: Upstream provider rejected the API key: {exc.message}",
            ) from exc
        except (APITimeoutError, APIConnectionError) as exc:
            logger.error("Upstream connection error during classification: %s", exc)
            # Default to CODING on transient classification error
            return "CODING"
        except Exception as exc:
            logger.warning("Classification exception, falling back to CODING: %s", exc)
            return "CODING"

    async def complete(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Perform non-streaming completion."""
        # Filter out None kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **clean_kwargs,
            )
            return response.model_dump()
        except AuthenticationError as exc:
            logger.error("Upstream authentication error: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"bad_upstream_api_key: Upstream provider rejected the API key: {exc.message}",
            ) from exc
        except APITimeoutError as exc:
            logger.error("Upstream timeout error: %s", exc)
            raise HTTPException(
                status_code=504,
                detail=f"upstream_timeout: Request to upstream model timed out: {exc}",
            ) from exc
        except APIConnectionError as exc:
            logger.error("Upstream connection error: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"upstream_connection_error: Unable to connect to upstream provider: {exc}",
            ) from exc
        except APIError as exc:
            logger.error("Upstream API error: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"upstream_api_error: {exc.message}",
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error in complete: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"internal_server_error: {str(exc)}",
            ) from exc

    async def stream_complete(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Perform streaming completion."""
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **clean_kwargs,
            )
            async for chunk in stream:
                yield chunk.model_dump()
        except AuthenticationError as exc:
            logger.error("Upstream authentication error during stream: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"bad_upstream_api_key: Upstream provider rejected the API key: {exc.message}",
            ) from exc
        except (APITimeoutError, APIConnectionError, APIError) as exc:
            logger.error("Upstream error during stream: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"upstream_stream_error: {str(exc)}",
            ) from exc
