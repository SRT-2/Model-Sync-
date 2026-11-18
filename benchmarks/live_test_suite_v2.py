"""Live end-to-end test harness for ModelSync Evaluation V2.

The paired experiment sends each of the same 80 prompts to:
(A) one direct OpenRouter baseline model, and
(B) the running ModelSync FastAPI server (classifier + selected downstream model).

Paid calls are never made unless both explicit CLI consent flags are supplied.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

from benchmarks.openrouter_v2 import OpenRouterClient, reconcile_generation
from benchmarks.v2_runner import (
    CATEGORIES, create_run_manifest, extract_actual_cost,
    extract_classifier_metrics, generation_data, load_dataset, load_json,
    make_price_snapshot, resolve_models, sha256_file, write_json,
)

ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("live_test_suite_v2")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:160]


def append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


async def reconcile(
    client: OpenRouterClient, run_dir: Path, generation_id: Optional[str],
    prompt_id: str, call_role: str, usage: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    result = await reconcile_generation(client, generation_id, attempts=4, delay_seconds=0.75)
    record = result.get("record")
    if generation_id and record is not None:
        write_json(run_dir / "raw_generation_records" / f"{stem(generation_id)}.json", record)
    usage_value = usage or {}
    return {
        "generation_id": generation_id,
        "generation_record": record,
        "reconciliation_status": result.get("status"),
        "reconciliation_error": result.get("error"),
        "actual_cost_usd": extract_actual_cost({
            "generation_record": record, "usage": usage_value
        }),
        "usage": usage_value,
        "call_role": call_role,
        "prompt_id": prompt_id,
    }


async def post_router(
    http: httpx.AsyncClient, run_dir: Path, prompt_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await http.post("/v1/chat/completions", json=payload)
        elapsed = round((time.perf_counter() - started) * 1000.0, 3)
        try:
            body: Any = response.json()
        except ValueError:
            body = {"raw_text": response.text}
        evidence = {
            "prompt_id": prompt_id, "request": payload,
            "http_status": response.status_code, "latency_ms": elapsed,
            "response_headers": {
                k: v for k, v in response.headers.items()
                if k.lower() in {"content-type", "x-routed-category", "x-routed-model", "request-id"}
            },
            "response_body": body,
        }
        write_json(run_dir / "raw_completions" / f"{stem(prompt_id)}_modelsync_api.json", evidence)
        return {
            "ok": response.is_success and isinstance(body, dict),
            "http_status": response.status_code, "latency_ms": elapsed,
            "body": body, "error": None if response.is_success else str(body)[:2000],
        }
    except Exception as exc:
        evidence = {
            "prompt_id": prompt_id, "request": payload,
            "exception_type": type(exc).__name__, "error": str(exc),
        }
        write_json(run_dir / "raw_errors" / f"{stem(prompt_id)}_modelsync_api.json", evidence)
        return {
            "ok": False, "http_status": None,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "body": {}, "error": str(exc),
        }


async def baseline_call(
    client: OpenRouterClient, run_dir: Path, prompt_id: str,
    model_id: str, messages: List[Dict[str, str]], max_tokens: int,
) -> Dict[str, Any]:
    started_at = now()
    started = time.perf_counter()
    try:
        result = await client.complete(model_id, messages, temperature=None, max_tokens=max_tokens)
        accounting = await reconcile(
            client, run_dir, result.get("generation_id"), prompt_id,
            "baseline_completion", result.get("usage"),
        )
        body = result.get("response") or {}
        record = {
            "prompt_id": prompt_id, "experiment": "baseline",
            "call_role": "baseline_completion", "status": "success",
            "started_at_utc": started_at, "requested_model": model_id,
            "actual_model": result.get("actual_model") or model_id,
            "provider_name": (generation_data(accounting["generation_record"]).get("provider_name")
                              or result.get("provider")),
            "http_status": result.get("http_status"),
            "latency_ms": result.get("latency_ms"),
            "generation_id": result.get("generation_id"),
            "usage": result.get("usage") or {},
            "actual_cost_usd": accounting.get("actual_cost_usd"),
            "reconciliation_status": accounting.get("reconciliation_status"),
            "finish_reason": result.get("finish_reason"),
            "response_text": result.get("content") or "",
        }
        append_jsonl(run_dir / "generation_records.jsonl", record)
        write_json(run_dir / "raw_completions" / f"{stem(prompt_id)}_baseline.json", {
            "prompt_id": prompt_id,
            "request": {
                "model": model_id, "messages": messages, "max_tokens": max_tokens,
                "temperature": None, "stream": False, "usage": {"include": True},
            },
            "http_status": result.get("http_status"),
            "latency_ms": result.get("latency_ms"),
            "generation_id": result.get("generation_id"),
            "response_body": body,
            "generation_record": accounting.get("generation_record"),
        })
        LOG.info("BASELINE prompt=%s status=success model=%s generation=%s cost=%s",
                 prompt_id, record["actual_model"], record["generation_id"],
                 record["actual_cost_usd"])
        return record
    except Exception as exc:
        record = {
            "prompt_id": prompt_id, "experiment": "baseline",
            "call_role": "baseline_completion", "status": "failed",
            "started_at_utc": started_at, "requested_model": model_id,
            "generation_id": None, "latency_ms": round((time.perf_counter()-started)*1000, 3),
            "actual_cost_usd": None, "reconciliation_status": "not_available",
            "error_type": type(exc).__name__, "error": str(exc),
        }
        append_jsonl(run_dir / "generation_records.jsonl", record)
        write_json(run_dir / "raw_errors" / f"{stem(prompt_id)}_baseline.json", record)
        LOG.error("BASELINE prompt=%s failed: %s", prompt_id, exc)
        return record


async def router_generation_accounting(
    client: OpenRouterClient, run_dir: Path, prompt_id: str, role: str,
    generation_id: Optional[str], usage: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    item = await reconcile(client, run_dir, generation_id, prompt_id, role, usage)
    data = generation_data(item.get("generation_record"))
    record = {
        "prompt_id": prompt_id, "experiment": "modelsync", "call_role": role,
        "status": "success" if generation_id else "missing_generation_id",
        "generation_id": generation_id, "usage": usage or {},
        "actual_cost_usd": item.get("actual_cost_usd"),
        "reconciliation_status": item.get("reconciliation_status"),
        "provider_name": data.get("provider_name"), "actual_model": data.get("model"),
        "raw_generation_record": item.get("generation_record"),
    }
    append_jsonl(run_dir / "generation_records.jsonl", record)
    return record


async def paired_cost_comparison(
    dataset: List[Dict[str, Any]], baseline_client: OpenRouterClient,
    router_client: OpenRouterClient, router_http: httpx.AsyncClient,
    run_dir: Path, models: Dict[str, str], max_tokens: int, max_cost: float,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    spend_guard = 0.0
    stop_reason = "completed"

    for index, item in enumerate(dataset, start=1):
        prompt_id = item["id"]
        messages = [{"role": "user", "content": item["prompt"]}]
        LOG.info("PAIRED_COST %d/%d prompt=%s", index, len(dataset), prompt_id)

        baseline = await baseline_call(
            baseline_client, run_dir, prompt_id, models["baseline"], messages, max_tokens
        )
        baseline_cost = baseline.get("actual_cost_usd")
        if baseline_cost is not None:
            spend_guard += float(baseline_cost)
        if spend_guard >= max_cost:
            rows.append({
                "prompt_id": prompt_id, "category": item["category"],
                "baseline": baseline, "status": "stopped_after_baseline_budget",
            })
            stop_reason = "max_cost_budget_reached"
            break

        payload = {
            "model": "smart-router", "messages": messages,
            "max_tokens": max_tokens, "stream": False,
            "include_router_telemetry": True,
        }
        api_result = await post_router(router_http, run_dir, prompt_id, payload)
        body = api_result.get("body") or {}
        telemetry = body.get("router_telemetry") or {}
        classifier = telemetry.get("classifier") or {}

        classifier_record = await router_generation_accounting(
            router_client, run_dir, prompt_id, "classifier",
            classifier.get("generation_id"), classifier.get("usage") or {},
        ) if classifier else {
            "status": "missing_telemetry", "generation_id": None,
            "actual_cost_usd": None, "usage": {},
        }

        downstream_usage = body.get("usage") or {}
        downstream_id = body.get("id")
        downstream_record = await router_generation_accounting(
            router_client, run_dir, prompt_id, "downstream_completion",
            downstream_id, downstream_usage,
        ) if api_result["ok"] else {
            "status": "failed", "generation_id": None,
            "actual_cost_usd": None, "usage": {},
        }
        if classifier:
            write_json(
                run_dir / "raw_completions" / f"{stem(prompt_id)}_classifier.json",
                {
                    "prompt_id": prompt_id,
                    "classifier_telemetry": classifier,
                    "generation_record": classifier_record.get("raw_generation_record"),
                    "reconciliation_status": classifier_record.get("reconciliation_status"),
                },
            )

        classifier_cost = classifier_record.get("actual_cost_usd")
        downstream_cost = downstream_record.get("actual_cost_usd")
        router_total = (
            float(classifier_cost) + float(downstream_cost)
            if classifier_cost is not None and downstream_cost is not None else None
        )
        if router_total is not None:
            spend_guard += router_total

        choices = body.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        row = {
            "prompt_id": prompt_id, "category": item["category"],
            "difficulty": item["difficulty"],
            "predicted_category": classifier.get("category"),
            "classifier_raw_output": classifier.get("raw_output"),
            "classifier_fallback": classifier.get("fallback"),
            "classifier_model": classifier.get("model"),
            "classifier_generation_id": classifier.get("generation_id"),
            "classifier_usage": classifier.get("usage") or {},
            "classifier_cost_usd": classifier_cost,
            "classifier_reconciliation": classifier_record.get("reconciliation_status"),
            "classifier_latency_ms": classifier.get("latency_ms"),
            "routed_category": body.get("routed_category") or telemetry.get("routed_category"),
            "selected_model": body.get("routed_model") or telemetry.get("routed_model"),
            "model_selection_correct": (
                body.get("routed_category") == item["category"]
                and classifier.get("fallback") is False
                and (body.get("routed_model") or telemetry.get("routed_model")) == {
                    "CODING": models["coding"],
                    "UNIT_TESTING": models["unit_testing"],
                    "REASONING": models["reasoning"],
                }.get(item["category"])
            ),
            "baseline_model": baseline.get("actual_model"),
            "baseline_generation_id": baseline.get("generation_id"),
            "baseline_usage": baseline.get("usage") or {},
            "baseline_cost_usd": baseline_cost,
            "baseline_latency_ms": baseline.get("latency_ms"),
            "downstream_model": downstream_record.get("actual_model") or body.get("model"),
            "downstream_generation_id": downstream_id,
            "downstream_usage": downstream_usage,
            "downstream_cost_usd": downstream_cost,
            "downstream_reconciliation": downstream_record.get("reconciliation_status"),
            "downstream_latency_ms": api_result.get("latency_ms"),
            "downstream_response": message.get("content") or "",
            "router_http_status": api_result.get("http_status"),
            "router_error": api_result.get("error"),
            "modelsync_total_cost_usd": router_total,
            "status": (
                "success" if baseline.get("status") == "success" and api_result["ok"]
                else "failed"
            ),
        }
        rows.append(row)
        write_json(run_dir / "paired_cost_in_progress.json", rows)

        if spend_guard >= max_cost:
            stop_reason = "max_cost_budget_reached"
            break

    metric_rows = [
        {
            "expected_category": row["category"],
            "classifier_prediction": (
                "INVALID" if row.get("classifier_fallback")
                else row.get("predicted_category") or "INVALID"
            ),
        }
        for row in rows if "category" in row and "predicted_category" in row
    ]
    classification = extract_classifier_metrics(metric_rows)
    baseline_costs = [row.get("baseline_cost_usd") for row in rows]
    router_costs = [row.get("modelsync_total_cost_usd") for row in rows]
    complete = (
        len(rows) == len(dataset)
        and all(value is not None for value in baseline_costs)
        and all(value is not None for value in router_costs)
        and all(row.get("status") == "success" for row in rows)
        and all(row.get("baseline", {}).get("reconciliation_status") == "reconciled" for row in rows)
        and all(row.get("classifier_reconciliation") == "reconciled" for row in rows)
        and all(row.get("downstream_reconciliation") == "reconciled" for row in rows)
    )
    baseline_total = sum(float(x) for x in baseline_costs if x is not None)
    router_total = sum(float(x) for x in router_costs if x is not None)
    savings = (
        round((baseline_total-router_total)/baseline_total*100.0, 4)
        if complete and baseline_total > 0 else None
    )
    result = {
        "prompts_expected": len(dataset), "prompts_compared": len(rows),
        "comparison_complete": complete, "stop_reason": stop_reason,
        "baseline_model": models["baseline"], "modelsync_models": models,
        "baseline_actual_cost_usd": round(baseline_total, 10),
        "modelsync_classifier_actual_cost_usd": round(
            sum(float(row["classifier_cost_usd"]) for row in rows
                if row.get("classifier_cost_usd") is not None), 10),
        "modelsync_downstream_actual_cost_usd": round(
            sum(float(row["downstream_cost_usd"]) for row in rows
                if row.get("downstream_cost_usd") is not None), 10),
        "modelsync_total_actual_cost_usd": round(router_total, 10),
        "actual_savings_percentage": savings,
        "actual_costs_complete": complete,
        "classifier_cost_included": True,
        "classification_metrics": classification,
        "correct_model_selections": sum(1 for row in rows if row.get("model_selection_correct")),
        "model_selection_accuracy": round(
            sum(1 for row in rows if row.get("model_selection_correct"))/len(rows), 4
        ) if rows else None,
        "note": "Actual costs come from OpenRouter generation records. Savings is withheld for incomplete or unreconciled paired results.",
    }
    write_json(run_dir / "paired_cost_results.json", {"summary": result, "records": rows})
    flat = []
    for row in rows:
        flat.append({
            "prompt_id": row.get("prompt_id"), "category": row.get("category"),
            "difficulty": row.get("difficulty"), "predicted_category": row.get("predicted_category"),
            "classifier_fallback": row.get("classifier_fallback"),
            "routed_category": row.get("routed_category"), "selected_model": row.get("selected_model"),
            "model_selection_correct": row.get("model_selection_correct"),
            "baseline_model": row.get("baseline_model"),
            "baseline_generation_id": row.get("baseline_generation_id"),
            "baseline_prompt_tokens": (row.get("baseline_usage") or {}).get("prompt_tokens"),
            "baseline_completion_tokens": (row.get("baseline_usage") or {}).get("completion_tokens"),
            "baseline_cost_usd": row.get("baseline_cost_usd"),
            "classifier_model": row.get("classifier_model"),
            "classifier_generation_id": row.get("classifier_generation_id"),
            "classifier_prompt_tokens": (row.get("classifier_usage") or {}).get("prompt_tokens"),
            "classifier_completion_tokens": (row.get("classifier_usage") or {}).get("completion_tokens"),
            "classifier_cost_usd": row.get("classifier_cost_usd"),
            "downstream_model": row.get("downstream_model"),
            "downstream_generation_id": row.get("downstream_generation_id"),
            "downstream_prompt_tokens": (row.get("downstream_usage") or {}).get("prompt_tokens"),
            "downstream_completion_tokens": (row.get("downstream_usage") or {}).get("completion_tokens"),
            "downstream_cost_usd": row.get("downstream_cost_usd"),
            "modelsync_total_cost_usd": row.get("modelsync_total_cost_usd"),
            "status": row.get("status"), "router_http_status": row.get("router_http_status"),
        })
    save_csv(run_dir / "paired_cost_results.csv", flat)
    return result


async def api_tests(
    http: httpx.AsyncClient, run_dir: Path, dataset: List[Dict[str, Any]],
    auth_enabled: bool,
) -> List[Dict[str, Any]]:
    tests: List[Dict[str, Any]] = []

    async def run(name: str, method: str, path: str, expected: List[int],
                  payload: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None) -> None:
        started = time.perf_counter()
        try:
            response = await http.request(method, path, json=payload, headers=headers)
            try:
                body: Any = response.json()
            except ValueError:
                body = {"raw_text": response.text}
            passed = response.status_code in expected
            record = {
                "test_name": name, "method": method, "path": path,
                "http_status": response.status_code, "expected_status": expected,
                "latency_ms": round((time.perf_counter()-started)*1000, 3),
                "passed": passed, "response": body,
            }
        except Exception as exc:
            record = {
                "test_name": name, "method": method, "path": path,
                "http_status": None, "expected_status": expected,
                "latency_ms": round((time.perf_counter()-started)*1000, 3),
                "passed": False, "error": str(exc),
            }
        tests.append(record)
        LOG.info("API_TEST %s status=%s passed=%s", name, record.get("http_status"), record["passed"])

    await run("GET health", "GET", "/health", [200])
    await run("GET models", "GET", "/v1/models", [200])
    if auth_enabled:
        await run("missing inbound auth", "GET", "/v1/models", [401], headers={"Authorization": ""})
        await run("invalid inbound auth", "GET", "/v1/models", [401],
                  headers={"Authorization": "Bearer invalid-evaluation-token"})

    for category in CATEGORIES:
        item = next(p for p in dataset if p["category"] == category)
        await run(
            f"live non-streaming {category}", "POST", "/v1/chat/completions", [200],
            payload={
                "model": "smart-router",
                "messages": [{"role": "user", "content": item["prompt"]}],
                "max_tokens": 100, "include_router_telemetry": True,
            },
        )
    await run("missing messages validation", "POST", "/v1/chat/completions", [422],
              payload={"model": "smart-router"})
    await run("invalid role validation", "POST", "/v1/chat/completions", [422],
              payload={"model": "smart-router", "messages": [{"role": "bad-role", "content": "x"}]})

    write_json(run_dir / "api_test_results.json", tests)
    save_csv(run_dir / "api_test_results.csv", tests)
    return tests


async def streaming_tests(
    http: httpx.AsyncClient, run_dir: Path, dataset: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records = []
    for category in CATEGORIES:
        item = next(p for p in dataset if p["category"] == category)
        payload = {
            "model": "smart-router",
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": 120, "stream": True, "include_router_telemetry": True,
        }
        started = time.perf_counter()
        first_event_ms = None
        events: List[str] = []
        status = None
        error = None
        try:
            async with http.stream("POST", "/v1/chat/completions", json=payload) as response:
                status = response.status_code
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        if first_event_ms is None:
                            first_event_ms = round((time.perf_counter()-started)*1000, 3)
                        events.append(line[5:].strip())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = round((time.perf_counter()-started)*1000, 3)
        parsed = []
        for event in events:
            if event and event != "[DONE]":
                try:
                    parsed.append(json.loads(event))
                except ValueError:
                    pass
        first = parsed[0] if parsed else {}
        telemetry = first.get("router_telemetry") or {}
        output = "".join(
            (chunk.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            for chunk in parsed
        )
        record = {
            "category": category, "prompt_id": item["id"],
            "http_status": status, "first_event_latency_ms": first_event_ms,
            "total_stream_latency_ms": elapsed, "event_count": len(events),
            "done_marker_received": "[DONE]" in events,
            "routed_category": telemetry.get("routed_category"),
            "routed_model": telemetry.get("routed_model"),
            "classifier_generation_id": (telemetry.get("classifier") or {}).get("generation_id"),
            "output_characters": len(output), "error": error,
            "passed": status == 200 and "[DONE]" in events and bool(output),
        }
        records.append(record)
        write_json(run_dir / "streaming" / f"{category.lower()}.json", {
            "request": payload, "events": events, "parsed_events": parsed, "summary": record,
        })
        LOG.info("STREAM %s status=%s done=%s", category, status, record["done_marker_received"])
    write_json(run_dir / "streaming_results.json", records)
    save_csv(run_dir / "streaming_results.csv", records)
    return records


async def performance_tests(
    http: httpx.AsyncClient, run_dir: Path, dataset: List[Dict[str, Any]],
    requests_per_level: int,
) -> List[Dict[str, Any]]:
    summaries = []
    for concurrency in (1, 5, 10):
        semaphore = asyncio.Semaphore(concurrency)
        chosen = [dataset[i % len(dataset)] for i in range(requests_per_level)]

        async def one(index: int, item: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                started = time.perf_counter()
                response = await post_router(
                    http, run_dir, f"PERF_C{concurrency}_{index:03d}",
                    {
                        "model": "smart-router",
                        "messages": [{"role": "user", "content": item["prompt"]}],
                        "max_tokens": 80, "include_router_telemetry": True,
                    },
                )
                return {
                    "request_index": index, "prompt_id": item["id"],
                    "category": item["category"], "success": response["ok"],
                    "http_status": response["http_status"],
                    "latency_ms": round((time.perf_counter()-started)*1000, 3),
                    "error": response["error"],
                }

        batch_started = time.perf_counter()
        batch = await asyncio.gather(*[
            one(i, item) for i, item in enumerate(chosen, 1)
        ])
        duration = time.perf_counter() - batch_started
        latency = sorted(r["latency_ms"] for r in batch if r["success"])

        def percentile(values: List[float], ratio: float) -> Optional[float]:
            if not values:
                return None
            at = (len(values)-1)*ratio
            low = int(at)
            high = min(low+1, len(values)-1)
            frac = at-low
            return round(values[low]*(1-frac)+values[high]*frac, 3)

        result = {
            "concurrency": concurrency, "requests": len(batch),
            "successful": len(latency), "failed": len(batch)-len(latency),
            "duration_seconds": round(duration, 3),
            "throughput_rps": round(len(latency)/duration, 4) if duration else 0,
            "mean_latency_ms": round(sum(latency)/len(latency), 3) if latency else None,
            "p50_latency_ms": percentile(latency, .50),
            "p95_latency_ms": percentile(latency, .95),
            "min_latency_ms": min(latency) if latency else None,
            "max_latency_ms": max(latency) if latency else None,
        }
        summaries.append(result)
        write_json(run_dir / "performance" / f"concurrency_{concurrency}.json",
                   {"summary": result, "requests": batch})
        LOG.info("PERFORMANCE concurrency=%d success=%d/%d rps=%s",
                 concurrency, len(latency), len(batch), result["throughput_rps"])
    write_json(run_dir / "performance_results.json", summaries)
    save_csv(run_dir / "performance_results.csv", summaries)
    return summaries


async def reliability_tests(
    router_http: httpx.AsyncClient, router_client: OpenRouterClient,
    run_dir: Path,
) -> List[Dict[str, Any]]:
    results = []
    health_response = await router_http.get("/health")
    health = health_response.json() if health_response.is_success else {}
    if health.get("auth_enabled"):
        for name, auth in [
            ("missing_router_auth", ""),
            ("malformed_router_auth", "Basic wrong-format"),
            ("invalid_router_auth", "Bearer invalid-evaluation-token"),
        ]:
            try:
                resp = await router_http.get("/v1/models", headers={"Authorization": auth})
                results.append({
                    "scenario": name, "type": "live_router_authentication",
                    "http_status": resp.status_code, "expected_status": 401,
                    "passed": resp.status_code == 401, "response": resp.text[:1000],
                })
            except Exception as exc:
                results.append({
                    "scenario": name, "type": "live_router_authentication",
                    "passed": False, "error": str(exc),
                })
    else:
        results.append({
            "scenario": "router_auth_disabled",
            "type": "configuration_observation", "passed": None,
            "note": "ROUTER_API_KEY is not enabled in the running server.",
        })

    try:
        await router_client.complete(
            "modelsync-evaluation/nonexistent-model",
            [{"role": "user", "content": "negative test"}],
            temperature=None, max_tokens=2,
        )
        results.append({
            "scenario": "invalid_upstream_model_id", "type": "live_upstream_negative_test",
            "passed": False, "note": "Provider unexpectedly accepted the invalid model ID.",
        })
    except Exception as exc:
        results.append({
            "scenario": "invalid_upstream_model_id", "type": "live_upstream_negative_test",
            "passed": True, "http_status": getattr(exc, "status_code", None),
            "error_type": type(exc).__name__, "error": str(exc),
        })

    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:1", timeout=1) as dead:
            await dead.get("/models")
        results.append({
            "scenario": "controlled_connection_failure",
            "type": "controlled_local_fault_injection", "passed": False,
        })
    except Exception as exc:
        results.append({
            "scenario": "controlled_connection_failure",
            "type": "controlled_local_fault_injection", "passed": True,
            "error_type": type(exc).__name__, "error": str(exc),
            "note": "This validates evidence capture for a local connection failure; it is not an OpenRouter outage test.",
        })

    write_json(run_dir / "reliability_results.json", results)
    save_csv(run_dir / "reliability_results.csv", results)
    return results


async def connectivity_tests(
    client: OpenRouterClient, run_dir: Path, models: Dict[str, str],
) -> List[Dict[str, Any]]:
    results = []
    for role, model in models.items():
        try:
            call = await client.complete(
                model, [{"role": "user", "content": "Reply with READY."}],
                temperature=None, max_tokens=8,
            )
            accounting = await reconcile(
                client, run_dir, call.get("generation_id"),
                f"CONNECT_{role}", role, call.get("usage"),
            )
            results.append({
                "role": role, "requested_model": model,
                "actual_model": call.get("actual_model"),
                "http_status": call.get("http_status"),
                "latency_ms": call.get("latency_ms"),
                "generation_id": call.get("generation_id"),
                "reconciliation_status": accounting.get("reconciliation_status"),
                "actual_cost_usd": accounting.get("actual_cost_usd"),
                "output": call.get("content"), "passed": bool(call.get("content")),
            })
        except Exception as exc:
            results.append({
                "role": role, "requested_model": model, "passed": False,
                "error_type": type(exc).__name__, "error": str(exc),
            })
    write_json(run_dir / "connectivity_results.json", results)
    save_csv(run_dir / "connectivity_results.csv", results)
    return results


def markdown_report(run_id: str, summary: Dict[str, Any]) -> str:
    paired = summary.get("paired_cost_comparison") or {}
    lines = [
        "# ModelSync Evaluation V2 — Live Test Report",
        "",
        f"- Run ID: {run_id}",
        f"- Created: {summary.get('created_at_utc')}",
        f"- Provider: OpenRouter",
        f"- Phases executed: {', '.join(summary.get('phases_executed', []))}",
        "",
        "## Paired cost comparison",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Prompts compared | {paired.get('prompts_compared', 'Not run')} |",
        f"| Comparison complete | {paired.get('comparison_complete', 'Not run')} |",
        f"| Baseline actual cost (USD) | {paired.get('baseline_actual_cost_usd', 'Not run')} |",
        f"| ModelSync classifier cost (USD) | {paired.get('modelsync_classifier_actual_cost_usd', 'Not run')} |",
        f"| ModelSync downstream cost (USD) | {paired.get('modelsync_downstream_actual_cost_usd', 'Not run')} |",
        f"| ModelSync total cost (USD) | {paired.get('modelsync_total_actual_cost_usd', 'Not run')} |",
        f"| Actual savings (%) | {paired.get('actual_savings_percentage', 'Not reportable')} |",
        "",
        "Savings is only reportable when all 80 paired prompts complete successfully and all baseline, classifier and downstream generation costs are reconciled.",
        "",
        "## Evidence",
        "",
        "See the run directory for raw outputs, generation records, run logs, API results, pricing snapshot, manifest, and checksums.",
        "",
        "## Scope notes",
        "",
        "Only phases listed as executed in summary.json have results. Controlled fault injection is labelled separately from real upstream failures.",
    ]
    return "\n".join(lines) + "\n"


async def execute(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    dataset_path = (ROOT / config["dataset"]["path"]).resolve()
    dataset = load_dataset(dataset_path, config["dataset"])
    models = resolve_models(config)

    baseline_key = os.getenv(config["secrets"]["baseline_api_key_env"]) or os.getenv(
        config["secrets"]["api_key_env"]
    )
    router_key = os.getenv(config["secrets"]["router_api_key_env"]) or os.getenv(
        config["secrets"]["api_key_env"]
    )
    if not baseline_key or not router_key:
        raise ValueError("Set OPENROUTER_API_KEY or separate baseline/router OpenRouter API keys.")

    run_id = f"live_v2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOG.addHandler(file_handler)
    LOG.setLevel(logging.INFO)

    baseline_client = OpenRouterClient(
        baseline_key, config["provider"]["base_url"],
        float(config["provider"].get("request_timeout_seconds", 120)),
        bool(config["provider"].get("allow_fallbacks", False)),
    )
    router_client = OpenRouterClient(
        router_key, config["provider"]["base_url"],
        float(config["provider"].get("request_timeout_seconds", 120)),
        bool(config["provider"].get("allow_fallbacks", False)),
    )
    headers = {"Content-Type": "application/json"}
    router_api_key = os.getenv("ROUTER_API_KEY")
    if router_api_key:
        headers["Authorization"] = f"Bearer {router_api_key}"
    router_http = httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), headers=headers,
        timeout=httpx.Timeout(args.api_timeout),
    )
    phases: Dict[str, Any] = {}

    try:
        write_json(run_dir / "inputs" / "evaluation_v2.json", config)
        write_json(run_dir / "inputs" / "dataset_v2_80.json", dataset)
        catalogue = await router_client.fetch_models()
        snapshot = make_price_snapshot(catalogue, models)
        write_json(run_dir / "pricing_snapshot.json", snapshot)
        manifest = create_run_manifest(
            run_id, config_path, dataset_path, config, dataset, models, snapshot
        )
        manifest.update({
            "api_base_url": args.base_url,
            "test_scope": [
                "direct model connectivity", "live FastAPI API integration",
                "paired same-prompt baseline versus ModelSync cost comparison",
                "live streaming", "live performance/concurrency",
                "live and controlled reliability checks",
            ],
            "paid_call_acknowledgement": True,
        })
        write_json(run_dir / "run_manifest.json", manifest)
        LOG.info("LIVE V2 START run=%s git=%s", run_id, manifest["source_provenance"].get("commit"))

        selected = set(args.phase)
        if "all" in selected:
            selected = {"connectivity", "api", "paired-cost", "streaming", "performance", "reliability"}

        if "connectivity" in selected:
            phases["connectivity"] = await connectivity_tests(router_client, run_dir, models)
        if "api" in selected:
            health_response = await router_http.get("/health")
            health = health_response.json() if health_response.is_success else {}
            phases["api"] = await api_tests(router_http, run_dir, dataset, bool(health.get("auth_enabled")))
        if "paired-cost" in selected:
            phases["paired_cost_comparison"] = await paired_cost_comparison(
                dataset, baseline_client, router_client, router_http, run_dir,
                models, args.max_tokens, args.max_cost_usd,
            )
        if "streaming" in selected:
            phases["streaming"] = await streaming_tests(router_http, run_dir, dataset)
        if "performance" in selected:
            phases["performance"] = await performance_tests(
                router_http, run_dir, dataset, args.performance_requests
            )
        if "reliability" in selected:
            phases["reliability"] = await reliability_tests(router_http, router_client, run_dir)

        summary = {
            "run_id": run_id, "created_at_utc": now(),
            "provider": "OpenRouter", "execution_mode": "live",
            "api_base_url": args.base_url, "dataset_count": len(dataset),
            "models": models, "phases_executed": sorted(selected),
            "phase_results": {
                key: (value if key == "paired_cost_comparison" else {
                    "total": len(value),
                    "passed": sum(1 for item in value if item.get("passed") is True),
                    "failed": sum(1 for item in value if item.get("passed") is False),
                })
                for key, value in phases.items()
            },
            "paired_cost_comparison": phases.get("paired_cost_comparison"),
            "api_keys_saved": False, "evidence_directory": str(run_dir),
        }
        write_json(run_dir / "summary.json", summary)
        (run_dir / "report.md").write_text(markdown_report(run_id, summary), encoding="utf-8")

        checksums = {}
        for file in sorted(run_dir.rglob("*")):
            if file.is_file() and file.name != "evidence_checksums.json":
                checksums[str(file.relative_to(run_dir))] = {
                    "sha256": sha256_file(file), "size_bytes": file.stat().st_size,
                }
        write_json(run_dir / "evidence_checksums.json", {
            "run_id": run_id, "created_at_utc": now(),
            "algorithm": "SHA-256", "files": checksums,
            "note": "Hashes detect changes; they are not a digital signature.",
        })
        LOG.info("LIVE V2 COMPLETE evidence_dir=%s", run_dir)
        return summary
    except Exception as exc:
        write_json(run_dir / "run_failure.json", {
            "run_id": run_id, "failed_at_utc": now(),
            "exception_type": type(exc).__name__, "error": str(exc),
            "completed_phases": sorted(phases),
        })
        LOG.exception("LIVE V2 ABORTED: %s", exc)
        raise
    finally:
        await baseline_client.close()
        await router_client.close()
        await router_http.aclose()
        file_handler.flush()
        file_handler.close()
        LOG.removeHandler(file_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live ModelSync Evaluation V2")
    parser.add_argument("--config", default=str(ROOT / "benchmarks" / "evaluation_v2.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "testing-results" / "live_evaluation_v2"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--phase", nargs="+",
        choices=["preflight", "connectivity", "api", "paired-cost", "streaming", "performance", "reliability", "all"],
        default=["all"],
    )
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--max-cost-usd", type=float, default=5.0)
    parser.add_argument("--performance-requests", type=int, default=5)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--i-understand-paid-calls", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(ROOT / ".env")

    if args.max_cost_usd <= 0 or args.performance_requests <= 0:
        parser.error("Budget and performance request count must be positive.")

    if args.phase == ["preflight"]:
        config = load_json(Path(args.config))
        dataset = load_dataset(ROOT / config["dataset"]["path"], config["dataset"])
        print(json.dumps({
            "dataset_count": len(dataset),
            "category_counts": {
                category: sum(1 for item in dataset if item["category"] == category)
                for category in CATEGORIES
            },
            "models": resolve_models(config),
            "api_base_url": args.base_url,
            "note": "Local-only preflight. No OpenRouter or ModelSync calls were made.",
        }, indent=2))
        return

    if not (args.live and args.i_understand_paid_calls):
        parser.error("Live phases require both --live and --i-understand-paid-calls.")

    result = asyncio.run(execute(args))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
