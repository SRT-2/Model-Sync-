"""Offline tests for live Evaluation V2 harness helpers."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmarks.live_test_suite_v2 import markdown_report, save_csv, stem


def test_live_harness_module_imports_without_network_calls() -> None:
    # Import itself validates the harness syntax and must not contact providers.
    assert callable(markdown_report)


def test_safe_stem_removes_path_and_control_characters() -> None:
    value = stem('../prompt 01/with spaces')
    assert '/' not in value
    assert ' ' not in value


def test_save_csv_writes_a_header_and_records(tmp_path: Path) -> None:
    target = tmp_path / 'nested' / 'results.csv'
    save_csv(target, [{'prompt_id': 'P001', 'cost_usd': 0.001},
                      {'prompt_id': 'P002', 'cost_usd': 0.002}])
    with target.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {'prompt_id': 'P001', 'cost_usd': '0.001'},
        {'prompt_id': 'P002', 'cost_usd': '0.002'},
    ]


def test_markdown_report_labels_savings_only_from_paired_summary() -> None:
    summary = {
        'created_at_utc': '2026-11-18T00:00:00+00:00',
        'phases_executed': ['paired-cost'],
        'paired_cost_comparison': {
            'prompts_compared': 80,
            'comparison_complete': True,
            'baseline_actual_cost_usd': 0.50,
            'modelsync_classifier_actual_cost_usd': 0.01,
            'modelsync_downstream_actual_cost_usd': 0.10,
            'modelsync_total_actual_cost_usd': 0.11,
            'actual_savings_percentage': 78.0,
        },
    }
    report = markdown_report('run-test', summary)
    assert '0.50' in report
    assert '0.11' in report
    assert '78.0' in report
    assert 'all 80 paired prompts' in report

def test_fault_injection_module_imports_without_network_calls() -> None:
    from benchmarks.fault_injection_v2 import FaultInjectionProvider

    provider = FaultInjectionProvider("upstream_timeout")
    assert provider.mode == "upstream_timeout"
    assert provider.classification_calls == 0

def test_fault_injection_upstream_timeout_returns_504() -> None:
    from benchmarks.fault_injection_v2 import execute_case

    record = execute_case(
        "test-timeout", "upstream_timeout",
        payload={
            "model": "smart-router",
            "messages": [{"role": "user", "content": "test"}],
            "include_router_telemetry": True,
        },
        expected_status=504,
        expected_substring="upstream_timeout",
    )
    assert record["passed"] is True
    assert record["http_status"] == 504


def test_fault_injection_mid_stream_records_sse_error() -> None:
    from benchmarks.fault_injection_v2 import execute_case

    record = execute_case(
        "test-stream-interruption", "stream_interrupted",
        payload={
            "model": "smart-router",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
            "include_router_telemetry": True,
        },
        expected_status=200,
        expected_substring="upstream_error",
    )
    assert record["passed"] is True
    assert record["http_status"] == 200

def test_router_health_preflight_rejects_model_mapping_mismatch() -> None:
    from benchmarks.live_test_suite_v2 import validate_router_health

    models = {
        "classifier": "classifier-id",
        "coding": "coding-id",
        "unit_testing": "testing-id",
        "reasoning": "reasoning-id",
        "baseline": "baseline-id",
    }
    health = {
        "status": "ok",
        "provider": "openrouter",
        "configured_models": {
            "coding": "wrong-coding-id",
            "unit_testing": "testing-id",
            "reasoning": "reasoning-id",
            "classifier": "classifier-id",
        },
    }
    with pytest.raises(ValueError, match="does not match"):
        validate_router_health(health, models)


def test_router_health_preflight_accepts_matching_mapping() -> None:
    from benchmarks.live_test_suite_v2 import validate_router_health

    models = {
        "classifier": "classifier-id",
        "coding": "coding-id",
        "unit_testing": "testing-id",
        "reasoning": "reasoning-id",
        "baseline": "baseline-id",
    }
    health = {
        "status": "ok",
        "provider": "openrouter",
        "configured_models": {
            "coding": "coding-id",
            "unit_testing": "testing-id",
            "reasoning": "reasoning-id",
            "classifier": "classifier-id",
        },
    }
    assert validate_router_health(health, models)["passed"] is True


def test_deferred_reconciler_uses_generation_record_cost() -> None:
    import asyncio
    from benchmarks.reconcile_live_run_v2 import fetch_one

    class FakeClient:
        async def get_generation(self, generation_id: str):
            return {"data": {
                "id": generation_id,
                "model": "vendor/model",
                "provider_name": "Provider",
                "total_cost": 0.000123,
            }}

    item = {
        "call_role": "classifier",
        "generation_id": "gen-test",
        "usage": {"cost": 0.000999},
    }
    result = asyncio.run(fetch_one(item, FakeClient(), FakeClient(), asyncio.Semaphore(1)))
    assert result["reconciliation_status"] == "reconciled"
    assert result["actual_cost_usd"] == pytest.approx(0.000123)
    assert result["provider_name"] == "Provider"


def test_deferred_reconciler_keeps_404_as_pending() -> None:
    import asyncio
    from benchmarks.reconcile_live_run_v2 import fetch_one

    class NotFoundError(Exception):
        status_code = 404

    class FakeClient:
        async def get_generation(self, generation_id: str):
            raise NotFoundError("not available yet")

    item = {
        "call_role": "classifier",
        "generation_id": "gen-pending",
        "usage": {"cost": 0.000005},
    }
    result = asyncio.run(fetch_one(item, FakeClient(), FakeClient(), asyncio.Semaphore(1)))
    assert result["reconciliation_status"] == "pending_not_found"
    assert result["actual_cost_usd"] is None
    assert result["usage"]["cost"] == 0.000005


def _paired_rows_and_records(pending_last: bool = False):
    rows = []
    generation_records = []
    for index in range(80):
        prompt_id = f"P{index:03d}"
        rows.append({
            "prompt_id": prompt_id,
            "category": "CODING",
            "difficulty": "clear",
            "status": "success",
            "baseline_generation_id": f"base-{index}",
            "classifier_generation_id": f"class-{index}",
            "downstream_generation_id": f"down-{index}",
            "baseline_cost_usd": 0.1,
            "classifier_cost_usd": 0.01,
            "downstream_cost_usd": 0.02,
            "baseline_reconciliation_status": "reconciled",
            "classifier_reconciliation": "reconciled",
            "downstream_reconciliation": "reconciled",
        })
        for role, cost in (
            ("baseline_completion", 0.1),
            ("classifier", 0.01),
            ("downstream_completion", 0.02),
        ):
            is_pending = pending_last and index == 79 and role == "classifier"
            generation_records.append({
                "prompt_id": prompt_id,
                "call_role": role,
                "generation_id": f"{role}-{index}",
                "usage": {"cost": cost},
                "actual_cost_usd": None if is_pending else cost,
                "reconciliation_status": "pending_not_found" if is_pending else "reconciled",
                "raw_generation_record": None if is_pending else {"data": {"total_cost": cost}},
            })
    return rows, generation_records


def test_deferred_paired_costs_withhold_savings_while_any_record_is_pending(tmp_path: Path) -> None:
    import json
    from benchmarks.reconcile_live_run_v2 import update_paired_costs

    rows, records = _paired_rows_and_records(pending_last=True)
    target = tmp_path / "paired_cost_results.json"
    target.write_text(json.dumps({"summary": {}, "records": rows}), encoding="utf-8")

    summary = update_paired_costs(tmp_path, records)

    assert summary["prompts_compared"] == 80
    assert summary["comparison_complete"] is False
    assert summary["actual_savings_percentage"] is None
    assert summary["actual_costs_complete"] is False


def test_deferred_paired_costs_report_savings_only_after_all_240_records_reconcile(tmp_path: Path) -> None:
    import json
    from benchmarks.reconcile_live_run_v2 import update_paired_costs

    rows, records = _paired_rows_and_records()
    target = tmp_path / "paired_cost_results.json"
    target.write_text(json.dumps({"summary": {}, "records": rows}), encoding="utf-8")

    summary = update_paired_costs(tmp_path, records)

    assert summary["prompts_compared"] == 80
    assert summary["comparison_complete"] is True
    assert summary["baseline_actual_cost_usd"] == pytest.approx(8.0)
    assert summary["modelsync_classifier_actual_cost_usd"] == pytest.approx(0.8)
    assert summary["modelsync_downstream_actual_cost_usd"] == pytest.approx(1.6)
    assert summary["modelsync_total_actual_cost_usd"] == pytest.approx(2.4)
    assert summary["actual_savings_percentage"] == pytest.approx(70.0)
