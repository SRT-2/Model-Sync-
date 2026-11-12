"""Comprehensive unit tests for the benchmark evaluation framework."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.config import Settings
from app.routing import CODING, UNIT_TESTING, REASONING, parse_label
from app.api.routes import select_target_model
from benchmarks.pricing import (
    calculate_cost,
    calculate_savings_percentage,
    lookup_model_price,
    ModelPrice,
)
from benchmarks.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_latency_percentiles,
)
from benchmarks.evaluator import (
    evaluate_response_quality,
    check_python_syntax,
    extract_code_blocks,
)
from benchmarks.runner import MockBenchmarkProvider, run_benchmark


# 1. Model Selection & Routing Tests
def test_select_target_model_coding():
    settings = Settings()
    model = select_target_model(CODING, settings)
    assert model == settings.coding_model


def test_select_target_model_testing():
    settings = Settings()
    model = select_target_model(UNIT_TESTING, settings)
    assert model == settings.testing_model


def test_select_target_model_reasoning():
    settings = Settings()
    model = select_target_model(REASONING, settings)
    assert model == settings.reasoning_model


def test_select_target_model_unknown_fallback():
    settings = Settings()
    model = select_target_model("UNKNOWN_CATEGORY", settings)
    # Unknown category falls back to coding_model safely
    assert model == settings.coding_model


# 2. Token-Cost Calculation & Pricing Tests
def test_token_cost_calculation():
    # Model: prompt=$0.60/M, completion=$2.40/M
    model_id = "nvidia/nemotron-3-ultra-550b-a55b"
    prompt_tokens = 1000
    completion_tokens = 500
    cost, status, price = calculate_cost(
        model_id, prompt_tokens, completion_tokens, use_paid_counterpart=False
    )
    # Expected: (1000 * 0.60 + 500 * 2.40) / 1,000,000 = (600 + 1200) / 1,000,000 = 0.0018
    assert round(cost, 6) == 0.0018
    assert status == "CALCULATED_CATALOG"
    assert price is not None


def test_token_cost_zero_for_free_model():
    model_id = "nvidia/nemotron-3-ultra-550b-a55b:free"
    cost, status, price = calculate_cost(
        model_id, 1000, 1000, use_paid_counterpart=False
    )
    assert cost == 0.0
    assert status == "MEASURED_FREE"


def test_token_cost_paid_counterpart_simulation():
    model_id = "nvidia/nemotron-3-ultra-550b-a55b:free"
    cost, status, price = calculate_cost(
        model_id, 1000, 1000, use_paid_counterpart=True
    )
    # 1000 * 0.60 + 1000 * 2.40 = 3.0 / 1000 = 0.003
    assert round(cost, 6) == 0.003
    assert status == "SIMULATED_PAID_CATALOG"


def test_zero_cost_baseline_savings_handling():
    # If baseline cost is 0.0, calculate_savings_percentage must not divide by zero
    savings = calculate_savings_percentage(baseline_cost=0.0, router_cost=0.0)
    assert savings is None


def test_positive_savings_percentage():
    baseline_cost = 1.00
    router_cost = 0.60
    savings = calculate_savings_percentage(baseline_cost, router_cost)
    assert savings == 40.0


def test_missing_model_price_fallback():
    cost, status, price = calculate_cost("nonexistent/unknown-model", 500, 200)
    assert cost == 0.0
    assert status == "PRICE_UNAVAILABLE"
    assert price is None


# 3. Classification Metrics & Confusion Matrix Tests
def test_classification_metrics_perfect_accuracy():
    actual = ["CODING", "UNIT_TESTING", "REASONING"]
    predicted = ["CODING", "UNIT_TESTING", "REASONING"]
    res = compute_classification_metrics(actual, predicted)
    assert res["accuracy"] == 1.0
    assert res["correct"] == 3
    assert res["macro_precision"] == 1.0
    assert res["macro_recall"] == 1.0
    assert res["macro_f1"] == 1.0


def test_classification_metrics_with_errors():
    actual = ["CODING", "UNIT_TESTING", "REASONING", "CODING"]
    predicted = ["CODING", "CODING", "REASONING", "CODING"]
    res = compute_classification_metrics(actual, predicted)
    assert res["accuracy"] == 0.75
    assert res["correct"] == 3
    # Check confusion matrix
    cm = res["confusion_matrix"]
    assert cm["UNIT_TESTING"]["CODING"] == 1
    assert cm["CODING"]["CODING"] == 2


def test_compute_latency_percentiles():
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    p = compute_latency_percentiles(latencies)
    assert p["count"] == 10
    assert p["mean"] == 5.5
    assert p["median"] == 5.5
    assert p["min"] == 1.0
    assert p["max"] == 10.0


def test_compute_latency_percentiles_empty():
    p = compute_latency_percentiles([])
    assert p["count"] == 0
    assert p["mean"] == 0.0


# 4. Evaluator & Syntax Checks
def test_python_syntax_checker():
    valid_code = "def foo():\n    return 42"
    valid, err = check_python_syntax(valid_code)
    assert valid is True
    assert err is None

    invalid_code = "def foo(\n return"
    valid, err = check_python_syntax(invalid_code)
    assert valid is False
    assert err is not None


def test_evaluator_empty_response():
    prompt_meta = {"category": "CODING", "rubric": {"required_elements": ["def"]}}
    res = evaluate_response_quality(prompt_meta, "")
    assert res["score"] == 0.0
    assert res["passed"] is False


def test_evaluator_valid_coding_response():
    prompt_meta = {
        "category": "CODING",
        "rubric": {
            "required_elements": ["class LRUCache", "def get", "def put", "capacity"],
            "max_score": 10,
        },
    }
    response_text = """Here is the solution:
```python
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
    def get(self, key):
        return -1
    def put(self, key, value):
        pass
```
This runs in O(1) time complexity with full docstrings and explanation."""
    res = evaluate_response_quality(prompt_meta, response_text)
    assert res["score"] >= 8.0
    assert res["passed"] is True
    assert res["syntax_valid"] is True


import asyncio

# 5. Benchmark Runner Serialization & Dry-Run Integration
def test_run_benchmark_dry_run_generates_valid_artifacts(tmp_path):
    summary = asyncio.run(
        run_benchmark(
            mode="smoke",
            live=False,
            output_dir=tmp_path,
            max_tokens=100,
        )
    )
    assert summary["dataset_size"] == 3
    assert "routing_performance" in summary
    assert "quality_performance" in summary
    assert "cost_analysis" in summary

    # Verify files created on disk
    run_id = summary["run_id"]
    summary_file = tmp_path / f"summary_{run_id}.json"
    records_csv = tmp_path / f"records_{run_id}.csv"
    records_json = tmp_path / f"records_{run_id}.json"

    assert summary_file.exists()
    assert records_csv.exists()
    assert records_json.exists()

    with open(summary_file, "r") as f:
        data = json.load(f)
        assert data["run_id"] == run_id
