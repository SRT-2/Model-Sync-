from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import HTTPException
from openai import (
    AsyncOpenAI,
    AuthenticationError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)

from app.config import Settings
from app.providers.base import LLMProvider
from app.routing import CLASSIFIER_SYSTEM_PROMPT, parse_label, CODING, UNIT_TESTING, REASONING

logger = logging.getLogger(__name__)


class TogetherProvider(LLMProvider):
    """OpenAI-compatible provider adapter (OpenRouter or Together AI).

    The legacy class name is retained to avoid changing existing imports.
    """

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

    async def classify_with_metadata(self, classifier_model: str, prompt: str) -> Dict[str, Any]:
        """Classify and optionally expose generation/usage telemetry to an opted-in caller."""
        messages = [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        started = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=classifier_model,
                messages=messages,
                temperature=0.0,
                max_tokens=self.settings.classifier_max_output_tokens,
                extra_body={
                    "usage": {"include": True},
                    "provider": {"allow_fallbacks": self.settings.openrouter_allow_fallbacks},
                } if self.settings.provider_name == "openrouter" else None,
            )
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            response_data = response.model_dump()
            choices = response_data.get("choices") or []
            raw_output = ""
            if choices:
                raw_output = ((choices[0].get("message") or {}).get("content") or "")
            normalized = raw_output.strip().upper()
            fallback = normalized not in {CODING, UNIT_TESTING, REASONING}
            return {
                "category": parse_label(raw_output),
                "raw_output": raw_output,
                "request_messages": messages,
                "raw_response": response_data,
                "model": response_data.get("model") or classifier_model,
                "generation_id": response_data.get("id"),
                "usage": response_data.get("usage") or {},
                "latency_ms": latency_ms,
                "fallback": fallback,
                "fallback_reason": "non_standard_classifier_output" if fallback else None,
                "metadata_available": True,
            }
        except AuthenticationError as exc:
            logger.error("Upstream authentication failed during classification: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"bad_upstream_api_key: Upstream provider rejected the API key: {exc.message}",
            ) from exc
        except (APITimeoutError, APIConnectionError) as exc:
            logger.error("Upstream connection error during classification: %s", exc)
            return {
                "category": CODING,
                "raw_output": None,
                "request_messages": messages,
                "model": classifier_model,
                "generation_id": None,
                "usage": {},
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "fallback": True,
                "fallback_reason": type(exc).__name__,
                "metadata_available": False,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Classification exception, falling back to CODING: %s", exc)
            return {
                "category": CODING,
                "raw_output": None,
                "model": classifier_model,
                "generation_id": None,
                "usage": {},
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "fallback": True,
                "fallback_reason": type(exc).__name__,
                "metadata_available": False,
            }

    async def classify(self, classifier_model: str, prompt: str) -> str:
        metadata = await self.classify_with_metadata(classifier_model, prompt)
        return str(metadata["category"])

    async def complete(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if self.settings.provider_name == "openrouter":
            clean_kwargs.setdefault("extra_body", {
                "usage": {"include": True},
                "provider": {"allow_fallbacks": self.settings.openrouter_allow_fallbacks},
            })
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
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if self.settings.provider_name == "openrouter":
            clean_kwargs.setdefault("extra_body", {
                "usage": {"include": True},
                "provider": {"allow_fallbacks": self.settings.openrouter_allow_fallbacks},
            })
            clean_kwargs.setdefault("stream_options", {"include_usage": True})
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
