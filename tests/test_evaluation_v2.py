"""Deterministic validation tests for Evaluation V2 preparation (no network/API calls)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.v2_runner import (
    estimate_cost_from_catalog,
    extract_classifier_metrics,
    extract_actual_cost,
    load_dataset,
    make_price_snapshot,
    parse_classifier_label,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v2_dataset_has_80_unique_balanced_prompts() -> None:
    config = json.loads((ROOT / "benchmarks" / "evaluation_v2.json").read_text(encoding="utf-8"))
    dataset = load_dataset(ROOT / config["dataset"]["path"], config["dataset"])
    assert len(dataset) == 80
    assert len({item["id"] for item in dataset}) == 80
    assert len({item["prompt"] for item in dataset}) == 80
    assert {category: sum(p["category"] == category for p in dataset)
            for category in ("CODING", "UNIT_TESTING", "REASONING")} == {
                "CODING": 27, "UNIT_TESTING": 27, "REASONING": 26
            }
    assert all(set(item) == {"id", "category", "difficulty", "prompt", "description"}
               for item in dataset)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CODING", "CODING"),
        ("UNIT_TESTING", "UNIT_TESTING"),
        ("REASONING", "REASONING"),
        ("coding ", "CODING"),
        ("The answer is CODING", None),
        ("CODING / REASONING", None),
        ("", None),
        ("UNKNOWN", None),
    ],
)
def test_classifier_output_is_validated_strictly(raw: str, expected: str | None) -> None:
    assert parse_classifier_label(raw) == expected


def test_invalid_classifier_output_counts_as_incorrect() -> None:
    records = [
        {"expected_category": "CODING", "classifier_prediction": "INVALID"},
        {"expected_category": "UNIT_TESTING", "classifier_prediction": "UNIT_TESTING"},
        {"expected_category": "REASONING", "classifier_prediction": "REASONING"},
    ]
    metrics = extract_classifier_metrics(records)
    assert metrics["correct_predictions"] == 2
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["invalid_classifier_outputs"] == 1


def test_pricing_snapshot_requires_exact_model_ids() -> None:
    catalogue = {
        "data": [
            {
                "id": "vendor/model-a",
                "name": "Model A",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "context_length": 32000,
                "supported_parameters": ["temperature"],
            }
        ]
    }
    models = {role: "vendor/model-a" for role in
              ("classifier", "coding", "unit_testing", "reasoning", "baseline")}
    snapshot = make_price_snapshot(catalogue, models)
    assert snapshot["models"]["baseline"]["pricing"]["prompt"] == "0.000001"

    models["baseline"] = "vendor/unknown"
    with pytest.raises(ValueError, match="not present in OpenRouter catalogue"):
        make_price_snapshot(catalogue, models)


def test_actual_cost_prefers_generation_record_to_response_usage() -> None:
    result = {
        "generation_record": {"data": {"total_cost": "0.0042"}},
        "usage": {"cost": 0.005},
    }
    assert extract_actual_cost(result) == pytest.approx(0.0042)


def test_catalog_estimate_is_separate_and_uses_per_token_catalogue_prices() -> None:
    snapshot = {
        "models": {
            "baseline": {
                "id": "vendor/model-a",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            }
        }
    }
    estimate = estimate_cost_from_catalog(
        "vendor/model-a",
        {"prompt_tokens": 100, "completion_tokens": 50},
        snapshot,
    )
    assert estimate == pytest.approx(0.0002)
