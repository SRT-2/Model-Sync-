from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator, Dict, List
from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.responses import StreamingResponse

from app.api.deps import verify_api_key
from app.config import Settings, get_settings
from app.routing import (
    CODING,
    UNIT_TESTING,
    REASONING,
    extract_last_user_content,
    render_sse_format,
)
from app.schemas import (
    ChatCompletionRequest,
    HealthResponse,
    ModelCard,
    ModelListResponse,
)
from app.state import get_state

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="2.0.0",
        provider=settings.provider_name,
        configured_models={
            "coding": settings.coding_model,
            "unit_testing": settings.testing_model,
            "reasoning": settings.reasoning_model,
            "classifier": settings.classifier_model,
        },
        auth_enabled=bool(settings.router_api_key and settings.router_api_key.strip()),
    )


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(
    _: None = Depends(verify_api_key),
) -> ModelListResponse:
    return ModelListResponse(
        object="list",
        data=[
            ModelCard(id="smart-router", object="model", created=1700000000, owned_by="smart-router")
        ],
    )


def select_target_model(category: str, settings: Settings) -> str:
    if category == UNIT_TESTING:
        return settings.testing_model
    elif category == REASONING:
        return settings.reasoning_model
    return settings.coding_model


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    response: Response,
    _: None = Depends(verify_api_key),
) -> Any:
    state = get_state()
    settings = state.settings

    # 1. Extract prompt to classify
    user_prompt = extract_last_user_content(request.messages)
    
    # 2. Classify intent
    if user_prompt:
        category = await state.provider.classify(settings.classifier_model, user_prompt)
    else:
        category = CODING

    # 3. Select target model
    target_model = select_target_model(category, settings)

    # 4. Convert messages for upstream
    formatted_messages = [
        {"role": m.role, "content": m.content, **({"name": m.name} if m.name else {})}
        for m in request.messages
    ]

    # Passthrough parameters
    kwargs: Dict[str, Any] = {}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.stop is not None:
        kwargs["stop"] = request.stop

    # Allow extra parameters passed by clients
    for key, value in request.model_extra.items() if request.model_extra else []:
        if key not in kwargs and key not in {"model", "messages", "stream"}:
            kwargs[key] = value

    if request.stream:
        async def event_generator() -> AsyncIterator[str]:
            # Initial metadata event so streaming clients immediately know routing decision
            init_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "smart-router",
                "routed_category": category,
                "routed_model": target_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield render_sse_format(init_chunk)

            try:
                async for chunk in state.provider.stream_complete(
                    target_model, formatted_messages, **kwargs
                ):
                    yield render_sse_format(chunk)
            except Exception as exc:
                err_chunk = {
                    "error": {
                        "message": str(exc),
                        "type": "upstream_error",
                    }
                }
                yield render_sse_format(err_chunk)
            
            yield render_sse_format("[DONE]")

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Routed-Category": category,
            "X-Routed-Model": target_model,
            "Access-Control-Expose-Headers": "X-Routed-Category, X-Routed-Model",
        }
        return StreamingResponse(event_generator(), headers=headers, media_type="text/event-stream")

    # Non-streaming execution
    result = await state.provider.complete(target_model, formatted_messages, **kwargs)

    # Attach backward-compatible routing metadata
    result["routed_category"] = category
    result["routed_model"] = target_model

    # Also set response headers
    response.headers["X-Routed-Category"] = category
    response.headers["X-Routed-Model"] = target_model
    response.headers["Access-Control-Expose-Headers"] = "X-Routed-Category, X-Routed-Model"

    return result
