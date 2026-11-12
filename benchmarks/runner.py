"""LLM Smart Router Evaluation Benchmark Runner.

Executes comparative evaluation between Baseline (monolithic model) and Smart Router.
Generates structured JSON and CSV results, metrics, confusion matrix, and cost audits.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings, get_settings
from app.providers.base import LLMProvider
from app.providers.together import TogetherProvider
from app.api.routes import select_target_model
from app.routing import CODING, UNIT_TESTING, REASONING

from benchmarks.pricing import (
    calculate_cost,
    calculate_savings_percentage,
    lookup_model_price,
    PRICE_CATALOG,
    PAID_COUNTERPART_MAPPING,
)
from benchmarks.metrics import (
    compute_classification_metrics,
    compute_latency_percentiles,
)
from benchmarks.evaluator import evaluate_response_quality

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


class MockBenchmarkProvider(LLMProvider):
    """Deterministic mock provider for dry-run and offline tests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def classify(self, classifier_model: str, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["test", "pytest", "jest", "fixture", "mock", "assert", "coverage"]):
            return UNIT_TESTING
        if any(w in prompt_lower for w in ["puzzle", "probability", "bayes", "trade-off", "architecture", "recurrence", "knight", "sizing"]):
            return REASONING
        return CODING

    async def complete(
        self, model: str, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        prompt = messages[-1].get("content", "") if messages else ""
        content = f"Mock response from {model} for prompt: {prompt[:40]}..."
        # If code or testing requested, include code blocks for rubric
        if "test" in prompt.lower():
            content = f"```python\ndef test_mock():\n    assert True\n```\nMock tests completed."
        elif "class" in prompt.lower() or "def " in prompt.lower():
            content = f"```python\ndef mock_solution():\n    return 42\n```\nMock code implementation."
        else:
            content = "Step 1: Analyzed premise.\nStep 2: Applied logical steps.\nStep 3: Conclusion reached."

        return {
            "id": f"chatcmpl-mock-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 85,
                "total_tokens": 205,
            },
        }

    async def stream_complete(
        self, model: str, messages: List[Dict[str, Any]], **kwargs: Any
    ):
        res = await self.complete(model, messages, **kwargs)
        yield res


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def run_benchmark(
    mode: str = "dry-run",
    live: bool = False,
    baseline_model: Optional[str] = None,
    dataset_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_requests: Optional[int] = None,
    max_cost_budget: float = 1.0,
    delay_between_requests: float = 1.0,
    max_tokens: int = 500,
    temperature: float = 0.0,
    coding_override: Optional[str] = None,
    testing_override: Optional[str] = None,
    reasoning_override: Optional[str] = None,
) -> Dict[str, Any]:
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    settings = get_settings()

    # Model overrides if specified
    if coding_override:
        settings.coding_model = coding_override
    if testing_override:
        settings.testing_model = testing_override
    if reasoning_override:
        settings.reasoning_model = reasoning_override

    effective_baseline = baseline_model or settings.reasoning_model

    ds_path = dataset_path or (PROJECT_ROOT / "benchmarks" / "dataset.json")
    out_dir = output_dir or (PROJECT_ROOT / "benchmarks" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(ds_path)
    if mode == "smoke":
        # Select 1 prompt per category
        smoke_prompts = []
        for cat in [CODING, UNIT_TESTING, REASONING]:
            found = next((p for p in dataset if p["category"] == cat), None)
            if found:
                smoke_prompts.append(found)
        dataset = smoke_prompts
    elif max_requests and max_requests < len(dataset):
        dataset = dataset[:max_requests]

    logger.info("Initializing Benchmark Run [%s]", run_id)
    logger.info("Mode: %s | Live Provider: %s", mode, live)
    logger.info("Dataset size: %d prompts", len(dataset))
    logger.info("Baseline Model: %s", effective_baseline)
    logger.info("Smart Router Targets -> CODING: %s | TESTING: %s | REASONING: %s",
                settings.coding_model, settings.testing_model, settings.reasoning_model)
    logger.info("Classifier Model: %s", settings.classifier_model)

    if live:
        if not settings.effective_api_key:
            raise ValueError("Live mode requested but no upstream API key found in configuration.")
        provider: LLMProvider = TogetherProvider(settings)
    else:
        provider = MockBenchmarkProvider(settings)

    records: List[Dict[str, Any]] = []
    actual_categories: List[str] = []
    predicted_categories: List[str] = []
    model_selections_correct: List[bool] = []

    cumulative_cost = 0.0

    for idx, item in enumerate(dataset):
        prompt_id = item["id"]
        expected_cat = item["category"]
        prompt_text = item["prompt"]
        actual_categories.append(expected_cat)

        logger.info("[%d/%d] Processing %s (%s)...", idx + 1, len(dataset), prompt_id, expected_cat)

        # 1. Routing / Intent Classification
        t0_class = time.perf_counter()
        try:
            pred_cat = await provider.classify(settings.classifier_model, prompt_text)
            pred_cat = pred_cat.strip()
        except Exception as e:
            logger.warning("Classification error on %s: %s", prompt_id, e)
            pred_cat = CODING
        t_classify = time.perf_counter() - t0_class
        predicted_categories.append(pred_cat)

        # Select target model via routing logic
        selected_model = select_target_model(pred_cat, settings)
        expected_model = select_target_model(expected_cat, settings)
        model_correct = (selected_model == expected_model)
        model_selections_correct.append(model_correct)

        messages = [{"role": "user", "content": prompt_text}]

        # 2. Baseline Model Execution
        t0_base = time.perf_counter()
        base_status = "SUCCESS"
        base_error = None
        base_res: Dict[str, Any] = {}
        try:
            base_res = await provider.complete(
                effective_baseline,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            base_status = "FAILED"
            base_error = str(e)
            logger.error("Baseline call failed on %s: %s", prompt_id, e)
        t_base = time.perf_counter() - t0_base

        # 3. Smart Router Selected Model Execution
        t0_router = time.perf_counter()
        router_status = "SUCCESS"
        router_error = None
        router_res: Dict[str, Any] = {}
        try:
            router_res = await provider.complete(
                selected_model,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            router_status = "FAILED"
            router_error = str(e)
            logger.error("Router call failed on %s: %s", prompt_id, e)
        t_router_model = time.perf_counter() - t0_router
        t_router_total = t_classify + t_router_model

        # Extract responses and tokens
        base_text = ""
        base_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if base_status == "SUCCESS":
            choices = base_res.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                base_text = msg.get("content") or msg.get("reasoning") or ""
            base_usage = base_res.get("usage", {}) or base_usage

        router_text = ""
        router_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if router_status == "SUCCESS":
            choices = router_res.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                router_text = msg.get("content") or msg.get("reasoning") or ""
            router_usage = router_res.get("usage", {}) or router_usage

        # Provider-reported costs vs pricing catalog calculations
        # Baseline costs
        base_prompt_tokens = base_usage.get("prompt_tokens", 0)
        base_comp_tokens = base_usage.get("completion_tokens", 0)
        base_actual_cost = float(base_usage.get("cost", 0.0) or 0.0)
        base_sim_cost, _, _ = calculate_cost(
            effective_baseline, base_prompt_tokens, base_comp_tokens, use_paid_counterpart=True
        )

        # Router costs
        r_prompt_tokens = router_usage.get("prompt_tokens", 0)
        r_comp_tokens = router_usage.get("completion_tokens", 0)
        r_actual_cost = float(router_usage.get("cost", 0.0) or 0.0)
        r_sim_cost, _, _ = calculate_cost(
            selected_model, r_prompt_tokens, r_comp_tokens, use_paid_counterpart=True
        )

        cumulative_cost += r_actual_cost + base_actual_cost

        # 4. Response Quality Evaluation
        eval_base = evaluate_response_quality(item, base_text) if base_status == "SUCCESS" else {"score": 0.0, "passed": False, "notes": base_error or "Failed"}
        eval_router = evaluate_response_quality(item, router_text) if router_status == "SUCCESS" else {"score": 0.0, "passed": False, "notes": router_error or "Failed"}

        rec = {
            "prompt_id": prompt_id,
            "category_expected": expected_cat,
            "category_predicted": pred_cat,
            "classification_correct": (expected_cat == pred_cat),
            "difficulty": item.get("difficulty", "clear"),
            "baseline_model": effective_baseline,
            "router_model": selected_model,
            "model_selection_correct": model_correct,
            # Latency (seconds)
            "latency_classify_s": round(t_classify, 4),
            "latency_baseline_s": round(t_base, 4),
            "latency_router_model_s": round(t_router_model, 4),
            "latency_router_total_s": round(t_router_total, 4),
            # Tokens & Status
            "baseline_status": base_status,
            "router_status": router_status,
            "baseline_prompt_tokens": base_prompt_tokens,
            "baseline_completion_tokens": base_comp_tokens,
            "baseline_total_tokens": base_prompt_tokens + base_comp_tokens,
            "router_prompt_tokens": r_prompt_tokens,
            "router_completion_tokens": r_comp_tokens,
            "router_total_tokens": r_prompt_tokens + r_comp_tokens,
            # Costs (USD)
            "baseline_actual_cost": base_actual_cost,
            "baseline_simulated_cost": round(base_sim_cost, 6),
            "router_actual_cost": r_actual_cost,
            "router_simulated_cost": round(r_sim_cost, 6),
            # Quality Evaluation
            "baseline_score": eval_base["score"],
            "baseline_passed": eval_base["passed"],
            "router_score": eval_router["score"],
            "router_passed": eval_router["passed"],
            "score_diff": round(eval_router["score"] - eval_base["score"], 2),
            # Text outputs (truncated for record overview)
            "baseline_response_sample": (base_text[:120] + "...") if len(base_text) > 120 else base_text,
            "router_response_sample": (router_text[:120] + "...") if len(router_text) > 120 else router_text,
            "baseline_full_response": base_text,
            "router_full_response": router_text,
            "eval_notes": eval_router.get("notes", ""),
        }
        records.append(rec)

        if live and delay_between_requests > 0:
            await asyncio.sleep(delay_between_requests)

        if cumulative_cost > max_cost_budget:
            logger.warning("Cost budget threshold reached ($%.4f > $%.4f). Stopping early.",
                           cumulative_cost, max_cost_budget)
            break

    # Summary Metrics Calculation
    class_metrics = compute_classification_metrics(actual_categories, predicted_categories)
    model_sel_accuracy = sum(1 for m in model_selections_correct if m) / len(model_selections_correct) if model_selections_correct else 0.0

    base_latencies = [r["latency_baseline_s"] for r in records if r["baseline_status"] == "SUCCESS"]
    router_latencies = [r["latency_router_total_s"] for r in records if r["router_status"] == "SUCCESS"]

    base_scores = [r["baseline_score"] for r in records]
    router_scores = [r["router_score"] for r in records]

    # Costs
    tot_base_actual = sum(r["baseline_actual_cost"] for r in records)
    tot_router_actual = sum(r["router_actual_cost"] for r in records)
    tot_base_sim = sum(r["baseline_simulated_cost"] for r in records)
    tot_router_sim = sum(r["router_simulated_cost"] for r in records)

    actual_savings_pct = calculate_savings_percentage(tot_base_actual, tot_router_actual)
    sim_savings_pct = calculate_savings_percentage(tot_base_sim, tot_router_sim)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "live": live,
        "dataset_size": len(records),
        "configuration": {
            "baseline_model": effective_baseline,
            "coding_model": settings.coding_model,
            "testing_model": settings.testing_model,
            "reasoning_model": settings.reasoning_model,
            "classifier_model": settings.classifier_model,
            "provider_name": settings.provider_name,
        },
        "routing_performance": {
            "classification_accuracy": class_metrics["accuracy"],
            "model_selection_correctness": round(model_sel_accuracy, 4),
            "macro_precision": class_metrics["macro_precision"],
            "macro_recall": class_metrics["macro_recall"],
            "macro_f1": class_metrics["macro_f1"],
            "confusion_matrix": class_metrics["confusion_matrix"],
            "category_metrics": class_metrics["categories"],
        },
        "quality_performance": {
            "baseline_mean_score": round(sum(base_scores) / len(base_scores), 2) if base_scores else 0.0,
            "router_mean_score": round(sum(router_scores) / len(router_scores), 2) if router_scores else 0.0,
            "baseline_pass_rate": round(sum(1 for r in records if r["baseline_passed"]) / len(records), 4) if records else 0.0,
            "router_pass_rate": round(sum(1 for r in records if r["router_passed"]) / len(records), 4) if records else 0.0,
            "mean_score_difference": round((sum(router_scores) - sum(base_scores)) / len(records), 2) if records else 0.0,
        },
        "latency_performance": {
            "baseline_latency": compute_latency_percentiles(base_latencies),
            "router_total_latency": compute_latency_percentiles(router_latencies),
        },
        "cost_analysis": {
            "actual_billed": {
                "baseline_total_usd": round(tot_base_actual, 6),
                "router_total_usd": round(tot_router_actual, 6),
                "savings_percentage": actual_savings_pct,
                "notes": "Verified actual upstream billed cost returned by provider."
            },
            "simulated_paid_catalog": {
                "baseline_total_usd": round(tot_base_sim, 6),
                "router_total_usd": round(tot_router_sim, 6),
                "savings_percentage": round(sim_savings_pct, 2) if sim_savings_pct is not None else None,
                "notes": "Simulated cost using official published pricing catalog (date: 2026-10-24)."
            },
        },
    }

    # Write output artifacts
    summary_path = out_dir / f"summary_{run_id}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    records_json_path = out_dir / f"records_{run_id}.json"
    with open(records_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    records_csv_path = out_dir / f"records_{run_id}.csv"
    if records:
        fieldnames = [
            "prompt_id", "category_expected", "category_predicted", "classification_correct",
            "difficulty", "baseline_model", "router_model", "model_selection_correct",
            "latency_classify_s", "latency_baseline_s", "latency_router_model_s", "latency_router_total_s",
            "baseline_status", "router_status",
            "baseline_prompt_tokens", "baseline_completion_tokens", "baseline_total_tokens",
            "router_prompt_tokens", "router_completion_tokens", "router_total_tokens",
            "baseline_actual_cost", "baseline_simulated_cost",
            "router_actual_cost", "router_simulated_cost",
            "baseline_score", "baseline_passed", "router_score", "router_passed", "score_diff",
        ]
        with open(records_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                writer.writerow(r)

    logger.info("Artifacts saved: \n  JSON: %s\n  CSV: %s\n  Summary: %s",
                records_json_path, records_csv_path, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Smart Router Evaluation Benchmark")
    parser.add_argument("--mode", choices=["dry-run", "smoke", "full"], default="dry-run",
                        help="Execution mode (dry-run=preflight mock, smoke=3 samples, full=all prompts)")
    parser.add_argument("--live", action="store_true", default=False,
                        help="Opt-in to execute real provider calls (requires valid API key)")
    parser.add_argument("--baseline-model", default=None,
                        help="Explicit model ID to use as the fixed-model baseline")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Path to benchmark dataset JSON file")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to save benchmark output artifacts")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="Limit maximum requests to evaluate")
    parser.add_argument("--max-cost", type=float, default=0.50,
                        help="Maximum dollar budget ceiling for live requests")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay in seconds between requests (rate limit throttle)")
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Maximum output tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature")
    parser.add_argument("--coding-model", default=None, help="Override configured CODING model")
    parser.add_argument("--testing-model", default=None, help="Override configured TESTING model")
    parser.add_argument("--reasoning-model", default=None, help="Override configured REASONING model")

    args = parser.parse_args()

    asyncio.run(run_benchmark(
        mode=args.mode,
        live=args.live,
        baseline_model=args.baseline_model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_requests=args.max_requests,
        max_cost_budget=args.max_cost,
        delay_between_requests=args.delay,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        coding_override=args.coding_model,
        testing_override=args.testing_model,
        reasoning_override=args.reasoning_model,
    ))


if __name__ == "__main__":
    main()
