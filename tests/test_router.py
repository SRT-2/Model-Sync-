from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.providers.base import LLMProvider
from app.routing import (
    CODING,
    UNIT_TESTING,
    REASONING,
    collect_stream_merges_deltas,
    extract_last_user_content,
    parse_label,
    render_sse_format,
)
from app.schemas import ChatMessage
from app.state import set_provider_override


class MockLLMProvider(LLMProvider):
    def __init__(self, classification: str = "CODING") -> None:
        self.classification = classification
        self.last_model_used: str | None = None
        self.last_messages: list[dict] | None = None

    async def classify(self, classifier_model: str, prompt: str) -> str:
        return self.classification

    async def complete(
        self, model: str, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        self.last_model_used = model
        self.last_messages = messages
        return {
            "id": "chatcmpl-mock-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Mock response from {model}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    async def stream_complete(
        self, model: str, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        self.last_model_used = model
        self.last_messages = messages
        chunks = ["Mock ", "streaming ", "from ", model]
        for c in chunks:
            yield {
                "id": "chatcmpl-mock-stream",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": c},
                        "finish_reason": None,
                    }
                ],
            }


@pytest.fixture
def mock_provider():
    provider = MockLLMProvider()
    set_provider_override(provider)
    return provider


@pytest.fixture
def client(mock_provider):
    app = create_app()
    return TestClient(app)


# 1. Label parsing tests
@pytest.mark.parametrize(
    "raw_input,expected",
    [
        (" coding ", CODING),
        ("**REASONING**", REASONING),
        ("", CODING),
        ("CODING", CODING),
        ("REASONING!", REASONING),
        ("The answer is UNIT_TESTING", UNIT_TESTING),
        ("UNIT_TESTING.", UNIT_TESTING),
        ("garbage", CODING),
    ],
)
def test_parse_label(raw_input: str, expected: str):
    assert parse_label(raw_input) == expected


# 2. Extract last user content
def test_extract_last_user_content():
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "First prompt"},
        {"role": "assistant", "content": "Assistant answer"},
        {"role": "user", "content": "Second user prompt"},
    ]
    assert extract_last_user_content(messages) == "Second user prompt"

    typed_messages = [
        ChatMessage(role="user", content="Typed user prompt"),
        ChatMessage(role="assistant", content="Assistant answer"),
    ]
    assert extract_last_user_content(typed_messages) == "Typed user prompt"


# 3. SSE formatting
def test_render_sse_format():
    done_msg = render_sse_format("[DONE]")
    assert done_msg == "data: [DONE]\n\n"

    dict_data = {"key": "val"}
    formatted = render_sse_format(dict_data)
    assert formatted == f"data: {json.dumps(dict_data)}\n\n"


# 4. Stream delta collection
def test_collect_stream_merges_deltas():
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": ", "}}]},
        {"choices": [{"delta": {"content": "World!"}}]},
    ]
    assert collect_stream_merges_deltas(chunks) == "Hello, World!"


# 5. Role validation rejects bad role
def test_validation_rejects_bad_role(client):
    payload = {
        "model": "smart-router",
        "messages": [
            {"role": "invalid_role", "content": "Hello"}
        ]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422


# 6. Health endpoint
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "configured_models" in data
    assert "coding" in data["configured_models"]


# 7. Models endpoint lists smart-router
def test_models_endpoint_lists_smart_router(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "smart-router" in ids


# 8. Auth required when configured
def test_auth_required_when_configured(monkeypatch):
    monkeypatch.setenv("ROUTER_API_KEY", "super-secret-key")
    # Clear settings cache
    get_settings.cache_clear()
    
    app = create_app()
    test_client = TestClient(app)

    # Missing header -> 401
    resp1 = test_client.get("/v1/models")
    assert resp1.status_code == 401

    # Bad token -> 401
    resp2 = test_client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert resp2.status_code == 401

    # Correct token -> 200
    resp3 = test_client.get("/v1/models", headers={"Authorization": "Bearer super-secret-key"})
    assert resp3.status_code == 200

    # Reset
    monkeypatch.delenv("ROUTER_API_KEY", raising=False)
    get_settings.cache_clear()


# 9. Chat completions routes and injects system prompt
def test_chat_completions_routes_and_injects_system_prompt(mock_provider, client):
    # Test routing to UNIT_TESTING
    mock_provider.classification = "UNIT_TESTING"
    settings = get_settings()

    payload = {
        "model": "smart-router",
        "messages": [
            {"role": "user", "content": "Write pytest unit tests for my login function"}
        ]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["routed_category"] == "UNIT_TESTING"
    assert data["routed_model"] == settings.testing_model
    assert response.headers["X-Routed-Category"] == "UNIT_TESTING"
    assert response.headers["X-Routed-Model"] == settings.testing_model
    assert mock_provider.last_model_used == settings.testing_model
