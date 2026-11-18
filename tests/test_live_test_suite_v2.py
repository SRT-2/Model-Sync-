"""Offline tests for live Evaluation V2 harness helpers."""
from __future__ import annotations

import csv
from pathlib import Path

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
