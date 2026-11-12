"""OpenRouter HTTP helpers and usage reconciliation for Evaluation V2.

This module makes real OpenRouter calls only when invoked by the live runner.
It does not contain or read API keys from source control.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import httpx


class OpenRouterRequestError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None,
                 payload: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str,
                 timeout_seconds: float = 120.0,
                 allow_fallbacks: bool = False) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("An OpenRouter API key is required.")
        self.base_url = base_url.rstrip("/")
        self.allow_fallbacks = allow_fallbacks
        self.http = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
                "X-Title": "ModelSync Evaluation V2"
            },
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def fetch_models(self) -> Dict[str, Any]:
        response = await self.http.get("/models")
        if response.is_error:
            raise OpenRouterRequestError(
                f"OpenRouter model catalogue returned HTTP {response.status_code}",
                response.status_code, _safe_json(response)
            )
        return response.json()

    async def complete(self, model: str, messages: list[dict[str, str]],
                       temperature: Optional[float], max_tokens: int) -> Dict[str, Any]:
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "usage": {"include": True},
            "provider": {"allow_fallbacks": self.allow_fallbacks},
        }
        # Some reasoning models do not accept temperature. Omit it unless
        # the catalogue explicitly lists it as a supported parameter.
        if temperature is not None:
            body["temperature"] = temperature
        started = time.perf_counter()
        response = await self.http.post("/chat/completions", json=body)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        payload = _safe_json(response)

        if response.is_error:
            error = payload.get("error", payload)
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise OpenRouterRequestError(
                f"OpenRouter returned HTTP {response.status_code}: {message}",
                response.status_code, payload
            )

        if not isinstance(payload, dict):
            raise OpenRouterRequestError("OpenRouter returned a non-object JSON response",
                                         response.status_code, {})
        if payload.get("error"):
            error = payload["error"]
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise OpenRouterRequestError(f"OpenRouter completion error: {message}",
                                         response.status_code, payload)

        choices = payload.get("choices") or []
        if not choices:
            raise OpenRouterRequestError("OpenRouter response has no choices",
                                         response.status_code, payload)
        choice = choices[0]
        if choice.get("error") or choice.get("finish_reason") == "error":
            raise OpenRouterRequestError(
                f"OpenRouter choice error: {choice.get('error') or 'finish_reason=error'}",
                response.status_code, payload
            )

        return {
            "response": payload,
            "http_status": response.status_code,
            "latency_ms": round(elapsed_ms, 3),
            "generation_id": payload.get("id"),
            "requested_model": model,
            "actual_model": payload.get("model"),
            "usage": payload.get("usage") or {},
            "provider": payload.get("provider"),
            "finish_reason": choice.get("finish_reason"),
            "content": ((choice.get("message") or {}).get("content") or ""),
        }

    async def get_generation(self, generation_id: str) -> Dict[str, Any]:
        response = await self.http.get("/generation", params={"id": generation_id})
        payload = _safe_json(response)
        if response.is_error:
            raise OpenRouterRequestError(
                f"Generation lookup returned HTTP {response.status_code}",
                response.status_code, payload
            )
        if not isinstance(payload, dict):
            raise OpenRouterRequestError("Generation lookup returned invalid JSON",
                                         response.status_code, {})
        return payload


async def reconcile_generation(client: OpenRouterClient, generation_id: Optional[str],
                               attempts: int = 3,
                               delay_seconds: float = 0.5) -> Dict[str, Any]:
    """Fetch the authoritative generation record after a completion."""
    if not generation_id:
        return {"status": "missing_generation_id", "record": None}
    last_error: Optional[str] = None
    for attempt in range(max(1, attempts)):
        try:
            record = await client.get_generation(generation_id)
            return {"status": "reconciled", "record": record}
        except Exception as exc:  # keep the original completion record even if lookup fails
            last_error = str(exc)
            if attempt + 1 < attempts:
                await asyncio.sleep(delay_seconds)
    return {"status": "reconciliation_failed", "error": last_error, "record": None}


def _safe_json(response: httpx.Response) -> Dict[str, Any]:
    try:
        value = response.json()
        return value if isinstance(value, dict) else {"_value": value}
    except ValueError:
        return {"_non_json_body": response.text[:4000]}
