"""Controlled local failure-injection tests for the ModelSync FastAPI routes.

These scenarios exercise HTTP/error handling with an in-process fault provider.
They do not contact OpenRouter and must not be described as real provider outages.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.config import get_settings
from app.providers.base import LLMProvider
from app.state import get_state, set_provider_override

ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("fault_injection_v2")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FaultInjectionProvider(LLMProvider):
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.classification_calls = 0

    async def classify(self, classifier_model: str, prompt: str) -> str:
        self.classification_calls += 1
        return "CODING"

    async def classify_with_metadata(self, classifier_model: str, prompt: str) -> Dict[str, Any]:
        self.classification_calls += 1
        if self.mode == "classifier_auth_failure":
            raise HTTPException(status_code=502, detail="bad_upstream_api_key: injected auth rejection")
        if self.mode == "classifier_timeout_fallback":
            return {
                "category": "CODING", "raw_output": None,
                "model": classifier_model, "generation_id": None, "usage": {},
                "latency_ms": 120000.0, "fallback": True,
                "fallback_reason": "APITimeoutError", "metadata_available": False,
            }
        if self.mode == "classifier_invalid_output":
            return {
                "category": "CODING", "raw_output": "UNEXPECTED_LABEL",
                "model": classifier_model, "generation_id": None, "usage": {},
                "latency_ms": 10.0, "fallback": True,
                "fallback_reason": "non_standard_classifier_output", "metadata_available": False,
            }
        return {
            "category": "CODING", "raw_output": "CODING",
            "model": classifier_model, "generation_id": None, "usage": {},
            "latency_ms": 5.0, "fallback": False, "fallback_reason": None,
            "metadata_available": False,
        }

    async def complete(
        self, model: str, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        if self.mode == "upstream_auth_failure":
            raise HTTPException(status_code=502, detail="bad_upstream_api_key: injected auth rejection")
        if self.mode == "upstream_rate_limit":
            raise HTTPException(status_code=502, detail="upstream_api_error: injected HTTP 429")
        if self.mode == "upstream_timeout":
            raise HTTPException(status_code=504, detail="upstream_timeout: injected timeout")
        if self.mode == "upstream_connection_failure":
            raise HTTPException(status_code=502, detail="upstream_connection_error: injected connection error")
        if self.mode == "upstream_server_error":
            raise HTTPException(status_code=502, detail="upstream_api_error: injected HTTP 500")
        prompt = messages[-1].get("content", "") if messages else ""
        return {
            "id": f"fault-injection-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"Controlled response for {prompt[:40]}"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    async def stream_complete(
        self, model: str, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        yield {
            "id": "fault-stream-id",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": "partial output"}, "finish_reason": None}],
        }
        if self.mode == "stream_interrupted":
            raise RuntimeError("Controlled mid-stream interruption")
        yield {
            "id": "fault-stream-id",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }


def configure(mode: str, router_api_key: str | None = None) -> FaultInjectionProvider:
    # Empty string overrides any ROUTER_API_KEY value from a local .env file
    # for scenarios that intentionally test unauthenticated access.
    os.environ["ROUTER_API_KEY"] = router_api_key or ""
    get_settings.cache_clear()
    state = get_state()
    state.reload()
    provider = FaultInjectionProvider(mode)
    set_provider_override(provider)
    return provider


def execute_case(
    scenario: str,
    mode: str,
    method: str = "POST",
    path: str = "/v1/chat/completions",
    payload: Dict[str, Any] | None = None,
    expected_status: int = 200,
    expected_substring: str | None = None,
    expected_category: str | None = None,
    auth_key: str | None = None,
    headers: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    original_router_key = os.environ.get("ROUTER_API_KEY")
    provider = configure(mode, auth_key)
    from app.main import create_app

    started = time.perf_counter()
    try:
        with TestClient(create_app()) as client:
            response = client.request(
                method,
                path,
                json=payload if method.upper() != "GET" else None,
                headers=headers or {},
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        body_text = response.text
        try:
            body: Any = response.json()
        except ValueError:
            body = {"raw_text": body_text}

        passed = response.status_code == expected_status
        if expected_substring:
            passed = passed and expected_substring in body_text
        if expected_category:
            passed = passed and isinstance(body, dict) and body.get("routed_category") == expected_category

        return {
            "scenario": scenario,
            "test_type": "controlled_local_fault_injection",
            "provider": "in-process FaultInjectionProvider",
            "http_status": response.status_code,
            "expected_status": expected_status,
            "latency_ms": elapsed_ms,
            "passed": passed,
            "classifier_calls": provider.classification_calls,
            "response": body,
            "note": "Simulated locally; not evidence of a real OpenRouter outage.",
        }
    except Exception as exc:
        return {
            "scenario": scenario,
            "test_type": "controlled_local_fault_injection",
            "provider": "in-process FaultInjectionProvider",
            "http_status": None,
            "expected_status": expected_status,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if original_router_key is None:
            os.environ.pop("ROUTER_API_KEY", None)
        else:
            os.environ["ROUTER_API_KEY"] = original_router_key
        get_settings.cache_clear()
        state = get_state()
        state.reload()


def run_suite(output_dir: Path) -> Dict[str, Any]:
    run_id = f"fault_v2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)

    payload = {
        "model": "smart-router",
        "messages": [{"role": "user", "content": "Write unit tests for a login function"}],
        "include_router_telemetry": True,
    }
    scenarios = [
        ("upstream_auth_failure", "upstream_auth_failure", 502, "bad_upstream_api_key", None),
        ("upstream_rate_limit_429", "upstream_rate_limit", 502, "upstream_api_error", None),
        ("upstream_timeout", "upstream_timeout", 504, "upstream_timeout", None),
        ("upstream_connection_failure", "upstream_connection_failure", 502, "upstream_connection_error", None),
        ("upstream_server_error_500", "upstream_server_error", 502, "upstream_api_error", None),
        ("classifier_timeout_fallback", "classifier_timeout_fallback", 200, None, "CODING"),
        ("classifier_invalid_output_fallback", "classifier_invalid_output", 200, None, "CODING"),
        ("classifier_auth_failure", "classifier_auth_failure", 502, "bad_upstream_api_key", None),
    ]

    results = []
    try:
        for scenario, mode, expected_status, expected_substring, expected_category in scenarios:
            results.append(execute_case(
                scenario, mode, payload=payload,
                expected_status=expected_status,
                expected_substring=expected_substring,
                expected_category=expected_category,
            ))

        results.append(execute_case(
            "stream_interrupted_mid_generation",
            "stream_interrupted",
            payload={**payload, "stream": True},
            expected_status=200,
            expected_substring="upstream_error",
        ))

        results.append(execute_case(
            "inbound_auth_missing_header",
            "normal", method="GET", path="/v1/models",
            expected_status=401,
            expected_substring="Missing Authorization header",
            auth_key="evaluation-test-secret",
        ))
        results.append(execute_case(
            "inbound_auth_malformed_header",
            "normal", method="GET", path="/v1/models",
            expected_status=401,
            expected_substring="Bearer",
            auth_key="evaluation-test-secret",
            headers={"Authorization": "Basic invalid-format"},
        ))
        results.append(execute_case(
            "inbound_auth_invalid_token",
            "normal", method="GET", path="/v1/models",
            expected_status=401,
            expected_substring="Invalid API key",
            auth_key="evaluation-test-secret",
            headers={"Authorization": "Bearer invalid-token"},
        ))

        original_router_key = os.environ.get("ROUTER_API_KEY")
        try:
            provider = configure("normal")
            from app.main import create_app
            with TestClient(create_app()) as client:
                client.post("/v1/chat/completions", json=payload)
                client.post("/v1/chat/completions", json=payload)
        finally:
            if original_router_key is None:
                os.environ.pop("ROUTER_API_KEY", None)
            else:
                os.environ["ROUTER_API_KEY"] = original_router_key
            get_settings.cache_clear()
            get_state().reload()
        results.append({
            "scenario": "identical_prompt_cache_observation",
            "test_type": "controlled_local_observation",
            "provider": "in-process FaultInjectionProvider",
            "http_status": 200,
            "expected_status": 200,
            "classifier_calls": provider.classification_calls,
            "passed": provider.classification_calls == 2,
            "observed": "Classifier called for both requests; no classifier cache is implemented.",
            "note": "Feature observation, not a failure-handling pass.",
        })

        summary = {
            "run_id": run_id,
            "created_at_utc": now(),
            "suite": "ModelSync V2 controlled local failure injection",
            "test_type": "offline_simulated_faults",
            "total_scenarios": len(results),
            "passed": sum(1 for row in results if row.get("passed") is True),
            "failed": sum(1 for row in results if row.get("passed") is False),
            "results": results,
            "warning": "These results validate local FastAPI error-handling paths with a simulated provider. They are not live OpenRouter outage measurements.",
        }
        (run_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "scenario", "test_type", "provider", "http_status",
                "expected_status", "latency_ms", "passed", "classifier_calls",
                "error_type", "error", "note",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        report = [
            "# ModelSync V2 — Controlled Failure-Injection Report",
            "",
            f"- Run ID: {run_id}",
            f"- Total scenarios: {summary['total_scenarios']}",
            f"- Passed: {summary['passed']}",
            f"- Failed: {summary['failed']}",
            "",
            "> These are controlled local simulations, not real OpenRouter outage tests.",
            "",
            "| Scenario | Status | Result |",
            "|---|---:|---|",
        ]
        report.extend(
            f"| {row['scenario']} | {row.get('http_status')} | {'PASS' if row.get('passed') else 'FAIL/OBSERVATION'} |"
            for row in results
        )
        (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

        checksums = {}
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.name != "checksums.json":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                checksums[str(path.relative_to(run_dir))] = {
                    "sha256": digest, "size_bytes": path.stat().st_size,
                }
        (run_dir / "checksums.json").write_text(
            json.dumps({"algorithm": "SHA-256", "files": checksums}, indent=2),
            encoding="utf-8",
        )
        LOG.info("Fault-injection suite completed: %s", run_dir)
        return summary
    finally:
        LOG.removeHandler(handler)
        handler.flush()
        handler.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ModelSync controlled local fault-injection suite")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "testing-results" / "fault_injection_v2"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_suite(Path(args.output_dir))
    print(json.dumps({
        "run_id": result["run_id"],
        "total_scenarios": result["total_scenarios"],
        "passed": result["passed"],
        "failed": result["failed"],
        "warning": result["warning"],
    }, indent=2))


if __name__ == "__main__":
    main()
