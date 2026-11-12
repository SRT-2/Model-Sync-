from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Union

from app.schemas import ChatMessage


CODING = "CODING"
UNIT_TESTING = "UNIT_TESTING"
REASONING = "REASONING"

CLASSIFIER_SYSTEM_PROMPT = """You are an intent classifier for an LLM routing engine.
Analyze the user prompt and classify it into EXACTLY ONE of the following categories:
- CODING: Code generation, implementation, debugging, bug fixing, refactoring, algorithms, script writing, syntax questions.
- UNIT_TESTING: Writing unit tests, test suites, test cases, assertions, mock setups, edge-case testing, pytest, jest, unittest.
- REASONING: System architecture, high-level design, math, logical deduction, explanation, open-ended discussions, strategy, essay writing.

Respond with ONLY the category label: CODING, UNIT_TESTING, or REASONING. Do not provide explanations."""


def parse_label(raw: str) -> str:
    """Extract category label from classifier response. Default to CODING on failure."""
    if not raw or not isinstance(raw, str):
        return CODING

    cleaned = raw.strip().upper()
    
    # Check for exact or contained keywords
    # Check UNIT_TESTING before TESTING to avoid partial clashes
    if "UNIT_TESTING" in cleaned:
        return UNIT_TESTING
    if "REASONING" in cleaned:
        return REASONING
    if "CODING" in cleaned:
        return CODING

    # Regex search for word boundary
    match = re.search(r"\b(CODING|UNIT_TESTING|REASONING)\b", cleaned)
    if match:
        return match.group(1)

    return CODING


def extract_last_user_content(messages: List[Union[ChatMessage, Dict[str, Any]]]) -> str:
    """Find the most recent user message and return its text content."""
    for msg in reversed(messages):
        role = msg.role if isinstance(msg, ChatMessage) else msg.get("role")
        if role == "user":
            content = msg.content if isinstance(msg, ChatMessage) else msg.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                return "\n".join(text_parts)
    return ""


def render_sse_format(data: Union[str, Dict[str, Any]]) -> str:
    """Format payload according to the Server-Sent Events (SSE) data format."""
    if data == "[DONE]":
        return "data: [DONE]\n\n"
    if isinstance(data, str):
        return f"data: {data}\n\n"
    return f"data: {json.dumps(data)}\n\n"


def collect_stream_merges_deltas(chunks: List[Dict[str, Any]]) -> str:
    """Merge delta content from SSE chunks into a complete string."""
    merged = []
    for chunk in chunks:
        choices = chunk.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content")
        if content:
            merged.append(content)
    return "".join(merged)
