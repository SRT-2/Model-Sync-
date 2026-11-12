"""Live paid-model comparison runner for ModelSync Evaluation V2.

Compares a single fixed baseline model against classifier + category-routed
models over the same fixed dataset. Response-quality scoring is intentionally
excluded. This runner performs paid OpenRouter API calls only when both
--live and --i-understand-paid-calls are supplied.
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
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from dotenv import load_dotenv

from app.routing import CLASSIFIER_SYSTEM_PROMPT, CODING, UNIT_TESTING, REASONING
from benchmarks.openrouter_v2 import (
    OpenRouterClient,
    OpenRouterRequestError,
    reconcile_generation,
)

LOGGER = logging.getLogger("evaluation_v2")
CATEGORIES = [CODING, UNIT_TESTING, REASONING]
MODEL_ENV = {
    "classifier": "OPENROUTER_CLASSIFIER_MODEL",
    "coding": "OPENROUTER_CODING_MODEL",
    "unit_testing": "OPENROUTER_UNIT_TESTING_MODEL",
    "reasoning": "OPENROUTER_REASONING_MODEL",
    "baseline": "OPENROUTER_BASELINE_MODEL",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset(path: Path, expected: Dict[str, Any]) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON array.")

    expected_total = int(expected.get("expected_total", 80))
    if len(dataset) != expected_total:
        raise ValueError(f"Expected {expected_total} prompts, found {len(dataset)}.")

    ids = [item.get("id") for item in dataset]
    prompts = [item.get("prompt") for item in dataset]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Dataset IDs must be present and unique.")
    if any(not isinstance(value, str) or not value.strip() for value in prompts):
        raise ValueError("Every prompt must contain non-empty text.")
    if expected.get("expected_unique_prompts", True) and len(set(prompts)) != len(prompts):
        raise ValueError("Dataset prompts must be unique.")

    actual_counts: Dict[str, int] = {}
    for item in dataset:
        category = item.get("category")
        if category not in CATEGORIES:
            raise ValueError(f"Unsupported category in {item.get('id')}: {category}")
        if item.get("difficulty") not in {"clear", "boundary", "complex"}:
            raise ValueError(f"Invalid difficulty in {item.get('id')}")
        actual_counts[category] = actual_counts.get(category, 0) + 1

    expected_counts = expected.get("expected_categories", {})
    if actual_counts != expected_counts:
        raise ValueError(f"Category counts mismatch. Expected {expected_counts}, got {actual_counts}.")
    return dataset


def resolve_models(config: Dict[str, Any]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for role, env_name in MODEL_ENV.items():
        configured = os.getenv(env_name) or config.get("models", {}).get(role)
        if not configured or not str(configured).strip():
            raise ValueError(
                f"Missing model ID for {role}. Set {env_name} or benchmarks/evaluation_v2.json."
            )
        values[role] = str(configured).strip()
    return values


def parse_classifier_label(raw: str) -> Optional[str]:
    """Strict classifier parser: extra prose or multiple labels are invalid."""
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().upper()
    return normalized if normalized in CATEGORIES else None


def make_price_snapshot(catalog: Dict[str, Any], models: Dict[str, str]) -> Dict[str, Any]:
    entries = catalog.get("data", [])
    by_id = {entry.get("id"): entry for entry in entries if entry.get("id")}
    selected: Dict[str, Any] = {}
    for role, model_id in models.items():
        entry = by_id.get(model_id)
        if not entry:
            raise ValueError(f"Model ID is not present in OpenRouter catalogue: {role}={model_id}")
        pricing = entry.get("pricing") or {}
        selected[role] = {
            "id": entry.get("id"),
            "name": entry.get("name"),
            "context_length": entry.get("context_length"),
            "supported_parameters": entry.get("supported_parameters"),
            "pricing": pricing,
        }
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GET https://openrouter.ai/api/v1/models",
        "price_unit": "per token as returned by OpenRouter",
        "models": selected,
        "catalogue_snapshot_is_not_a_billing_record": True,
    }


def _decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _first_decimal(*values: Any) -> Optional[Decimal]:
    for value in values:
        parsed = _decimal(value)
        if parsed is not None:
            return parsed
    return None


def generation_data(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    data = record.get("data")
    return data if isinstance(data, dict) else record


def extract_actual_cost(call_result: Dict[str, Any]) -> Optional[float]:
    stats = generation_data(call_result.get("generation_record"))
    response_usage = call_result.get("usage") or {}
    cost = _first_decimal(
        stats.get("total_cost"),
        stats.get("cost"),
        (stats.get("usage") or {}).get("cost") if isinstance(stats.get("usage"), dict) else None,
        response_usage.get("cost"),
    )
    return float(cost) if cost is not None else None


def estimate_cost_from_catalog(
    model_id: Optional[str], usage: Dict[str, Any], price_snapshot: Dict[str, Any]
) -> Optional[float]:
    """Separate catalogue estimate; never label this as actual billed spend."""
    if not model_id:
        return None
    entry = next(
        (item for item in price_snapshot.get("models", {}).values() if item.get("id") == model_id),
        None,
    )
    if not entry:
        return None
    pricing = entry.get("pricing") or {}
    prompt_price = _decimal(pricing.get("prompt"))
    completion_price = _decimal(pricing.get("completion"))
    prompt_tokens = _decimal(usage.get("prompt_tokens"))
    completion_tokens = _decimal(usage.get("completion_tokens"))
    if None in (prompt_price, completion_price, prompt_tokens, completion_tokens):
        return None
    return float(prompt_price * prompt_tokens + completion_price * completion_tokens)


def extract_classifier_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    matrix = {actual: {pred: 0 for pred in CATEGORIES + ["INVALID"]} for actual in CATEGORIES}
    valid_correct = 0
    for record in records:
        actual = record["expected_category"]
        predicted = record.get("classifier_prediction") or "INVALID"
        if predicted not in CATEGORIES:
            predicted = "INVALID"
        matrix[actual][predicted] += 1
        if predicted == actual:
            valid_correct += 1

    category_metrics: Dict[str, Any] = {}
    precisions, recalls, f1s = [], [], []
    for category in CATEGORIES:
        tp = matrix[category][category]
        fp = sum(matrix[other][category] for other in CATEGORIES if other != category)
        fn = sum(matrix[category][other] for other in CATEGORIES + ["INVALID"] if other != category)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        category_metrics[category] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(matrix[category].values()),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    total = len(records)
    return {
        "total_prompts": total,
        "correct_predictions": valid_correct,
        "accuracy": round(valid_correct / total, 4) if total else 0.0,
        "invalid_classifier_outputs": sum(matrix[c]["INVALID"] for c in CATEGORIES),
        "macro_precision": round(sum(precisions) / len(precisions), 4),
        "macro_recall": round(sum(recalls) / len(recalls), 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "confusion_matrix": matrix,
        "per_category": category_metrics,
        "fallbacks_to_coding_are_counted_as_invalid_classifier_predictions": True,
    }


async def call_and_account(
    client: OpenRouterClient,
    prompt_id: str,
    experiment: str,
    call_role: str,
    model_id: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    run_dir: Path,
    config: Dict[str, Any],
    price_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        result = await client.complete(model_id, messages, temperature, max_tokens)
        generation_id = result.get("generation_id")
        reconciliation = await reconcile_generation(
            client,
            generation_id,
            attempts=int(config["provider"].get("generation_reconcile_attempts", 3)),
            delay_seconds=float(config["provider"].get("generation_reconcile_delay_seconds", 0.5)),
        )
        result["generation_record"] = reconciliation.get("record")
        result["reconciliation_status"] = reconciliation.get("status")
        result["reconciliation_error"] = reconciliation.get("error")
        actual_cost = extract_actual_cost(result)
        actual_model = result.get("actual_model") or model_id
        estimate = estimate_cost_from_catalog(actual_model, result.get("usage") or {}, price_snapshot)
        record = {
            "run_id": run_dir.name,
            "prompt_id": prompt_id,
            "experiment": experiment,
            "call_role": call_role,
            "status": "success",
            "started_at_utc": started_at,
            "requested_model": model_id,
            "actual_model": actual_model,
            "provider_name": generation_data(result.get("generation_record")).get("provider_name")
                or result.get("provider"),
            "generation_id": generation_id,
            "http_status": result.get("http_status"),
            "latency_ms": result.get("latency_ms"),
            "finish_reason": result.get("finish_reason"),
            "usage_response": result.get("usage") or {},
            "actual_cost_usd": actual_cost,
            "actual_cost_source": (
                "generation_record" if result.get("generation_record") and
                _first_decimal(generation_data(result["generation_record"]).get("total_cost"),
                               generation_data(result["generation_record"]).get("cost")) is not None
                else "completion_response_usage" if (result.get("usage") or {}).get("cost") is not None
                else None
            ),
            "catalog_estimated_cost_usd": estimate,
            "reconciliation_status": reconciliation.get("status"),
        }
        raw_dir = run_dir / "raw_generation_records"
        raw_dir.mkdir(parents=True, exist_ok=True)
        if generation_id:
            raw_path = raw_dir / f"{generation_id}.json"
            with raw_path.open("w", encoding="utf-8") as handle:
                json.dump(result.get("generation_record") or {}, handle, indent=2)
        return {"record": record, "result": result}
    except Exception as exc:
        status_code = exc.status_code if isinstance(exc, OpenRouterRequestError) else None
        payload = exc.payload if isinstance(exc, OpenRouterRequestError) else {}
        return {
            "record": {
                "run_id": run_dir.name,
                "prompt_id": prompt_id,
                "experiment": experiment,
                "call_role": call_role,
                "status": "failed",
                "started_at_utc": started_at,
                "requested_model": model_id,
                "actual_model": None,
                "provider_name": None,
                "generation_id": None,
                "http_status": status_code,
                "latency_ms": None,
                "finish_reason": None,
                "usage_response": {},
                "actual_cost_usd": None,
                "actual_cost_source": None,
                "catalog_estimated_cost_usd": None,
                "reconciliation_status": "not_available",
                "error": str(exc),
                "error_payload": payload,
            },
            "result": None,
        }


def append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


async def run_live(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config)
    config = load_json(config_path)
    dataset_path = PROJECT_ROOT / config["dataset"]["path"]
    dataset = load_dataset(dataset_path, config["dataset"])
    models = resolve_models(config)

    base_key = os.getenv(config["secrets"]["baseline_api_key_env"]) or os.getenv(
        config["secrets"]["api_key_env"]
    )
    router_key = os.getenv(config["secrets"]["router_api_key_env"]) or os.getenv(
        config["secrets"]["api_key_env"]
    )
    if not base_key or not router_key:
        raise ValueError("Set OPENROUTER_API_KEY or separate baseline/router OpenRouter API keys.")

    run_id = f"evaluation_v2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out_root = Path(args.output_dir)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    provider_cfg = config["provider"]
    base_url = provider_cfg["base_url"]
    baseline_client = OpenRouterClient(
        base_key, base_url, float(provider_cfg["request_timeout_seconds"]),
        bool(provider_cfg.get("allow_fallbacks", False))
    )
    router_client = OpenRouterClient(
        router_key, base_url, float(provider_cfg["request_timeout_seconds"]),
        bool(provider_cfg.get("allow_fallbacks", False))
    )

    try:
        catalog = await router_client.fetch_models()
        snapshot = make_price_snapshot(catalog, models)
        (run_dir / "pricing_snapshot.json").write_text(
            json.dumps(snapshot, indent=2), encoding="utf-8"
        )

        generation_cfg = config["generation"]
        max_cost = float(args.max_cost_usd if args.max_cost_usd is not None
                         else config["execution"]["default_max_cost_usd"])
        if max_cost <= 0:
            raise ValueError("--max-cost-usd must be greater than zero.")
        delay = float(config["execution"].get("delay_between_prompts_seconds", 0.5))
        call_records: List[Dict[str, Any]] = []
        prompt_records: List[Dict[str, Any]] = []
        budget_spend = 0.0
        budget_unknown = False
        stop_reason = None

        def update_budget(call_record: Dict[str, Any]) -> None:
            nonlocal budget_spend, budget_unknown
            actual = call_record.get("actual_cost_usd")
            estimated = call_record.get("catalog_estimated_cost_usd")
            if actual is not None:
                budget_spend += float(actual)
            elif estimated is not None:
                # Conservative estimate used only for the run guard, not reported as billing.
                budget_spend += float(estimated) * 1.25
            elif call_record.get("status") == "success":
                budget_unknown = True

        for index, item in enumerate(dataset, start=1):
            prompt_id = item["id"]
            category = item["category"]
            messages = [{"role": "user", "content": item["prompt"]}]
            row: Dict[str, Any] = {
                "prompt_id": prompt_id,
                "expected_category": category,
                "difficulty": item["difficulty"],
            }

            # Baseline: one fixed model for this prompt.
            baseline = await call_and_account(
                baseline_client, prompt_id, "baseline", "baseline_completion",
                models["baseline"], messages, float(generation_cfg["temperature"]),
                int(generation_cfg["max_output_tokens"]), run_dir, config, snapshot
            )
            row["baseline"] = baseline["record"]
            call_records.append(baseline["record"])
            append_jsonl(run_dir / "generation_records.jsonl", baseline["record"])
            update_budget(baseline["record"])

            if budget_unknown or budget_spend >= max_cost:
                stop_reason = "cost_budget_reached_or_cost_unavailable"
                prompt_records.append(row)
                break

            # ModelSync: live classifier followed by the category-selected model.
            classifier_messages = [
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
            ]
            classifier = await call_and_account(
                router_client, prompt_id, "modelsync", "classifier",
                models["classifier"], classifier_messages,
                float(generation_cfg["classifier_temperature"]),
                int(generation_cfg["classifier_max_output_tokens"]),
                run_dir, config, snapshot
            )
            row["classifier"] = classifier["record"]
            call_records.append(classifier["record"])
            append_jsonl(run_dir / "generation_records.jsonl", classifier["record"])
            update_budget(classifier["record"])

            raw_label = ""
            if classifier["result"] is not None:
                raw_label = classifier["result"].get("content", "")
            predicted = parse_classifier_label(raw_label) if classifier["result"] else None
            routed_category = predicted or CODING
            selected_model = {
                CODING: models["coding"],
                UNIT_TESTING: models["unit_testing"],
                REASONING: models["reasoning"],
            }[routed_category]
            row["classifier_raw_output"] = raw_label
            row["classifier_prediction"] = predicted or "INVALID"
            row["routed_category"] = routed_category
            row["selected_model"] = selected_model
            row["classification_correct"] = predicted == category
            row["model_selection_correct"] = predicted is not None and selected_model == {
                CODING: models["coding"],
                UNIT_TESTING: models["unit_testing"],
                REASONING: models["reasoning"],
            }[category]

            if budget_unknown or budget_spend >= max_cost:
                stop_reason = "cost_budget_reached_or_cost_unavailable"
                prompt_records.append(row)
                break

            downstream = await call_and_account(
                router_client, prompt_id, "modelsync", "downstream_completion",
                selected_model, messages, float(generation_cfg["temperature"]),
                int(generation_cfg["max_output_tokens"]), run_dir, config, snapshot
            )
            row["downstream"] = downstream["record"]
            call_records.append(downstream["record"])
            append_jsonl(run_dir / "generation_records.jsonl", downstream["record"])
            update_budget(downstream["record"])
            prompt_records.append(row)

            LOGGER.info(
                "[%d/%d] %s expected=%s predicted=%s route=%s budget_spend_guard=$%.6f",
                index, len(dataset), prompt_id, category, row["classifier_prediction"],
                routed_category, budget_spend
            )
            if budget_unknown or budget_spend >= max_cost:
                stop_reason = "cost_budget_reached_or_cost_unavailable"
                break
            if delay > 0:
                await asyncio.sleep(delay)

        classifier_rows = [
            row for row in prompt_records if "classifier_prediction" in row
        ]
        classifier_metrics = extract_classifier_metrics(classifier_rows)
        baseline_costs = [r["baseline"].get("actual_cost_usd") for r in prompt_records if r.get("baseline")]
        router_costs = []
        for row in prompt_records:
            for role in ("classifier", "downstream"):
                if role in row:
                    router_costs.append(row[role].get("actual_cost_usd"))

        baseline_cost_complete = (
            len(baseline_costs) == len(prompt_records) and all(v is not None for v in baseline_costs)
        )
        router_call_count = sum(1 for row in prompt_records for role in ("classifier", "downstream") if role in row)
        router_cost_complete = len(router_costs) == router_call_count and all(v is not None for v in router_costs)
        baseline_total = sum(float(v) for v in baseline_costs if v is not None)
        router_total = sum(float(v) for v in router_costs if v is not None)
        matched_prompts_complete = (
            len(prompt_records) == len(dataset)
            and all("baseline" in row and "classifier" in row and "downstream" in row
                    for row in prompt_records)
        )
        savings = None
        if (matched_prompts_complete and baseline_cost_complete
                and router_cost_complete and baseline_total > 0):
            savings = round((baseline_total - router_total) / baseline_total * 100.0, 4)

        summary = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provider": "OpenRouter",
            "live_api_calls": True,
            "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
            "dataset_size_expected": len(dataset),
            "prompts_started": len(prompt_records),
            "prompts_with_classifier_results": len(classifier_rows),
            "models": models,
            "provider_routing": {
                "allow_fallbacks": bool(provider_cfg.get("allow_fallbacks", False)),
                "note": "Record actual provider/model from each generation; model fallbacks are disabled for controlled comparison."
            },
            "classification_metrics": classifier_metrics,
            "routing": {
                "model_selection_correct": sum(1 for row in classifier_rows if row.get("model_selection_correct")),
                "evaluated": len(classifier_rows),
                "accuracy": round(
                    sum(1 for row in classifier_rows if row.get("model_selection_correct")) / len(classifier_rows), 4
                ) if classifier_rows else None,
                "invalid_classifier_outputs_fall_back_to_coding": True,
            },
            "cost_analysis": {
                "baseline_actual_cost_usd": round(baseline_total, 10),
                "modelsync_actual_cost_usd": round(router_total, 10),
                "modelsync_classifier_cost_included": True,
                "baseline_cost_complete": baseline_cost_complete,
                "modelsync_cost_complete": router_cost_complete,
                "matched_80_prompt_comparison_complete": matched_prompts_complete,
                "actual_savings_percentage": savings,
                "catalogue_estimates_are_not_included_in_actual_cost": True,
                "budget_guard_spend_usd": round(budget_spend, 10),
                "budget_guard_used_estimates_with_25_percent_buffer_when_actual_cost_missing": True,
                "max_cost_usd": max_cost,
                "unknown_cost_detected": budget_unknown,
            },
            "execution": {
                "stop_reason": stop_reason or "completed",
                "response_quality_scoring": False,
                "streaming": False,
                "call_count": len(call_records),
                "successful_calls": sum(1 for call in call_records if call.get("status") == "success"),
                "failed_calls": sum(1 for call in call_records if call.get("status") != "success"),
                "generation_records_reconciled": sum(1 for call in call_records if call.get("reconciliation_status") == "reconciled"),
            },
        }

        (run_dir / "prompt_results.json").write_text(
            json.dumps(prompt_records, indent=2, default=str), encoding="utf-8"
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        csv_path = run_dir / "prompt_results.csv"
        fields = [
            "prompt_id", "expected_category", "difficulty", "classifier_prediction",
            "routed_category", "selected_model", "classification_correct",
            "model_selection_correct", "baseline_status", "baseline_model",
            "baseline_generation_id", "baseline_prompt_tokens", "baseline_completion_tokens",
            "baseline_actual_cost_usd", "baseline_catalog_estimate_usd", "baseline_reconciliation",
            "classifier_status", "classifier_model", "classifier_generation_id",
            "classifier_prompt_tokens", "classifier_completion_tokens", "classifier_actual_cost_usd",
            "classifier_catalog_estimate_usd", "classifier_reconciliation", "downstream_status",
            "downstream_model", "downstream_generation_id", "downstream_prompt_tokens",
            "downstream_completion_tokens", "downstream_actual_cost_usd",
            "downstream_catalog_estimate_usd", "downstream_reconciliation",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in prompt_records:
                base = row.get("baseline", {})
                classify = row.get("classifier", {})
                downstream = row.get("downstream", {})
                base_usage = base.get("usage_response") or {}
                classify_usage = classify.get("usage_response") or {}
                down_usage = downstream.get("usage_response") or {}
                writer.writerow({
                    "prompt_id": row.get("prompt_id"),
                    "expected_category": row.get("expected_category"),
                    "difficulty": row.get("difficulty"),
                    "classifier_prediction": row.get("classifier_prediction"),
                    "routed_category": row.get("routed_category"),
                    "selected_model": row.get("selected_model"),
                    "classification_correct": row.get("classification_correct"),
                    "model_selection_correct": row.get("model_selection_correct"),
                    "baseline_status": base.get("status"),
                    "baseline_model": base.get("actual_model") or base.get("requested_model"),
                    "baseline_generation_id": base.get("generation_id"),
                    "baseline_prompt_tokens": base_usage.get("prompt_tokens"),
                    "baseline_completion_tokens": base_usage.get("completion_tokens"),
                    "baseline_actual_cost_usd": base.get("actual_cost_usd"),
                    "baseline_catalog_estimate_usd": base.get("catalog_estimated_cost_usd"),
                    "baseline_reconciliation": base.get("reconciliation_status"),
                    "classifier_status": classify.get("status"),
                    "classifier_model": classify.get("actual_model") or classify.get("requested_model"),
                    "classifier_generation_id": classify.get("generation_id"),
                    "classifier_prompt_tokens": classify_usage.get("prompt_tokens"),
                    "classifier_completion_tokens": classify_usage.get("completion_tokens"),
                    "classifier_actual_cost_usd": classify.get("actual_cost_usd"),
                    "classifier_catalog_estimate_usd": classify.get("catalog_estimated_cost_usd"),
                    "classifier_reconciliation": classify.get("reconciliation_status"),
                    "downstream_status": downstream.get("status"),
                    "downstream_model": downstream.get("actual_model") or downstream.get("requested_model"),
                    "downstream_generation_id": downstream.get("generation_id"),
                    "downstream_prompt_tokens": down_usage.get("prompt_tokens"),
                    "downstream_completion_tokens": down_usage.get("completion_tokens"),
                    "downstream_actual_cost_usd": downstream.get("actual_cost_usd"),
                    "downstream_catalog_estimate_usd": downstream.get("catalog_estimated_cost_usd"),
                    "downstream_reconciliation": downstream.get("reconciliation_status"),
                })
        LOGGER.info("Evaluation V2 artifacts saved to %s", run_dir)
        return summary
    finally:
        await baseline_client.close()
        await router_client.close()


async def catalog_preflight(config_path: Path, output_dir: Path) -> Dict[str, Any]:
    config = load_json(config_path)
    models = resolve_models(config)
    url = config["provider"]["base_url"].rstrip("/") + "/models"
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        catalog = response.json()
    snapshot = make_price_snapshot(catalog, models)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "pricing_snapshot_preflight.json"
    target.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return {"verified": list(snapshot["models"]), "snapshot": str(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelSync Evaluation V2 (OpenRouter)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "benchmarks" / "evaluation_v2.json"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "testing-results" / "evaluation_v2"))
    parser.add_argument("--catalog-preflight", action="store_true",
                        help="Fetch the public model catalogue and snapshot configured model pricing; no inference calls.")
    parser.add_argument("--live", action="store_true", help="Allow real paid inference calls.")
    parser.add_argument("--i-understand-paid-calls", action="store_true",
                        help="Required acknowledgement before paid inference calls.")
    parser.add_argument("--max-cost-usd", type=float, default=None,
                        help="Hard run budget guard in USD; default comes from evaluation_v2.json.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    load_dotenv(PROJECT_ROOT / ".env")
    if args.catalog_preflight:
        result = asyncio.run(catalog_preflight(Path(args.config), Path(args.output_dir)))
        print(json.dumps(result, indent=2))
        return
    if not (args.live and args.i_understand_paid_calls):
        parser.error("Paid execution requires both --live and --i-understand-paid-calls.")
    result = asyncio.run(run_live(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
