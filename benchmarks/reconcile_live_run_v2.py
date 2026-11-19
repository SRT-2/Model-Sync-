"""Deferred OpenRouter generation reconciliation for Evaluation V2 runs.

Run this after OpenRouter has published the generation metadata. This script
only performs GET requests; it never creates new model generations.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from benchmarks.live_test_suite_v2 import markdown_report, save_csv
from benchmarks.openrouter_v2 import OpenRouterClient
from benchmarks.v2_runner import generation_data, sha256_file, write_json

ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def actual_cost(record: Optional[Dict[str, Any]], usage: Optional[Dict[str, Any]]) -> Optional[float]:
    data = generation_data(record)
    for value in (
        data.get("total_cost"),
        data.get("cost"),
        (data.get("usage") or {}).get("cost") if isinstance(data.get("usage"), dict) else None,
        (usage or {}).get("cost"),
    ):
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


async def fetch_one(
    item: Dict[str, Any],
    baseline_client: OpenRouterClient,
    router_client: OpenRouterClient,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    if item.get("reconciliation_status") == "reconciled" and item.get("raw_generation_record"):
        return item
    generation_id = item.get("generation_id")
    if not generation_id:
        item["reconciliation_status"] = "missing_generation_id"
        return item

    # Baseline generation records belong to the baseline API key when separate
    # keys are configured. All ModelSync/router calls use the router key.
    client = baseline_client if item.get("call_role") == "baseline_completion" else router_client
    async with semaphore:
        try:
            raw = await client.get_generation(generation_id)
            data = generation_data(raw)
            if not data:
                item["reconciliation_status"] = "invalid_generation_record"
                item["reconciliation_error"] = "Generation response did not contain a data object."
                return item
            item["raw_generation_record"] = raw
            item["reconciliation_status"] = "reconciled"
            item["reconciliation_error"] = None
            item["actual_cost_usd"] = actual_cost(raw, item.get("usage"))
            item["provider_name"] = data.get("provider_name") or item.get("provider_name")
            item["actual_model"] = data.get("model") or item.get("actual_model")
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            item["reconciliation_status"] = (
                "pending_not_found" if status_code == 404
                else f"lookup_error_{status_code}" if status_code
                else "lookup_error"
            )
            item["reconciliation_error"] = str(exc)
        return item


def rewrite_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in records),
        encoding="utf-8",
    )
    temp.replace(path)


def update_connectivity(run_dir: Path, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    path = run_dir / "connectivity_results.json"
    if not path.exists():
        return None
    rows = read_json(path)
    lookup = {(item.get("prompt_id"), item.get("call_role")): item for item in records}
    for row in rows:
        role = row.get("role")
        record = lookup.get((f"CONNECT_{role}", role))
        if not record:
            continue
        row["reconciliation_status"] = record.get("reconciliation_status")
        row["actual_cost_usd"] = record.get("actual_cost_usd")
        row["provider_name"] = record.get("provider_name")
        row["actual_model"] = record.get("actual_model") or row.get("actual_model")
    write_json(path, rows)
    save_csv(run_dir / "connectivity_results.csv", rows)
    return {
        "total": len(rows),
        "passed": sum(1 for row in rows if row.get("passed") is True),
        "failed": sum(1 for row in rows if row.get("passed") is False),
        "reconciled": sum(1 for row in rows if row.get("reconciliation_status") == "reconciled"),
        "pending": sum(1 for row in rows if row.get("reconciliation_status") != "reconciled"),
    }


def update_paired_costs(run_dir: Path, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    path = run_dir / "paired_cost_results.json"
    if not path.exists():
        return None
    payload = read_json(path)
    rows = payload.get("records") or []
    lookup = {(item.get("prompt_id"), item.get("call_role")): item for item in records}

    for row in rows:
        prompt_id = row.get("prompt_id")
        baseline = lookup.get((prompt_id, "baseline_completion"))
        classifier = lookup.get((prompt_id, "classifier"))
        downstream = lookup.get((prompt_id, "downstream_completion"))

        if baseline:
            row["baseline_reconciliation_status"] = baseline.get("reconciliation_status")
            row["baseline_cost_usd"] = baseline.get("actual_cost_usd")
            row["baseline_model"] = baseline.get("actual_model") or row.get("baseline_model")
        if classifier:
            row["classifier_reconciliation"] = classifier.get("reconciliation_status")
            row["classifier_cost_usd"] = classifier.get("actual_cost_usd")
            row["classifier_model"] = classifier.get("actual_model") or row.get("classifier_model")
        if downstream:
            row["downstream_reconciliation"] = downstream.get("reconciliation_status")
            row["downstream_cost_usd"] = downstream.get("actual_cost_usd")
            row["downstream_model"] = downstream.get("actual_model") or row.get("downstream_model")

        base_cost = row.get("baseline_cost_usd")
        classifier_cost = row.get("classifier_cost_usd")
        downstream_cost = row.get("downstream_cost_usd")
        row["modelsync_total_cost_usd"] = (
            float(classifier_cost) + float(downstream_cost)
            if classifier_cost is not None and downstream_cost is not None else None
        )

    all_reconciled = (
        len(rows) == 80
        and all(row.get("status") == "success" for row in rows)
        and all(row.get("baseline_cost_usd") is not None for row in rows)
        and all(row.get("classifier_cost_usd") is not None for row in rows)
        and all(row.get("downstream_cost_usd") is not None for row in rows)
        and all(row.get("baseline_reconciliation_status") == "reconciled" for row in rows)
        and all(row.get("classifier_reconciliation") == "reconciled" for row in rows)
        and all(row.get("downstream_reconciliation") == "reconciled" for row in rows)
    )
    baseline_total = sum(float(row["baseline_cost_usd"]) for row in rows if row.get("baseline_cost_usd") is not None)
    classifier_total = sum(float(row["classifier_cost_usd"]) for row in rows if row.get("classifier_cost_usd") is not None)
    downstream_total = sum(float(row["downstream_cost_usd"]) for row in rows if row.get("downstream_cost_usd") is not None)
    modelsync_total = classifier_total + downstream_total
    savings = (
        round((baseline_total - modelsync_total) / baseline_total * 100.0, 4)
        if all_reconciled and baseline_total > 0 else None
    )

    summary = payload.get("summary") or {}
    summary.update({
        "prompts_expected": 80,
        "prompts_compared": len(rows),
        "comparison_complete": all_reconciled,
        "stop_reason": "completed_after_deferred_reconciliation" if all_reconciled else "generation_records_still_pending_or_pairs_incomplete",
        "baseline_actual_cost_usd": round(baseline_total, 10),
        "modelsync_classifier_actual_cost_usd": round(classifier_total, 10),
        "modelsync_downstream_actual_cost_usd": round(downstream_total, 10),
        "modelsync_total_actual_cost_usd": round(modelsync_total, 10),
        "actual_savings_percentage": savings,
        "actual_costs_complete": all_reconciled,
        "classifier_cost_included": True,
        "note": "Final costs come from OpenRouter generation records. Savings is reportable only when all 80 prompt pairs and all 240 generation records are reconciled.",
    })
    payload["summary"] = summary
    payload["records"] = rows
    write_json(path, payload)

    flat = []
    for row in rows:
        flat.append({
            "prompt_id": row.get("prompt_id"),
            "category": row.get("category"),
            "difficulty": row.get("difficulty"),
            "predicted_category": row.get("predicted_category"),
            "classifier_fallback": row.get("classifier_fallback"),
            "routed_category": row.get("routed_category"),
            "selected_model": row.get("selected_model"),
            "model_selection_correct": row.get("model_selection_correct"),
            "baseline_model": row.get("baseline_model"),
            "baseline_generation_id": row.get("baseline_generation_id"),
            "baseline_prompt_tokens": (row.get("baseline_usage") or {}).get("prompt_tokens"),
            "baseline_completion_tokens": (row.get("baseline_usage") or {}).get("completion_tokens"),
            "baseline_cost_usd": row.get("baseline_cost_usd"),
            "baseline_reconciliation_status": row.get("baseline_reconciliation_status"),
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
            "status": row.get("status"),
            "router_http_status": row.get("router_http_status"),
        })
    save_csv(run_dir / "paired_cost_results.csv", flat)
    return summary


async def reconcile_run(run_dir: Path) -> Dict[str, Any]:
    config = read_json(run_dir / "inputs" / "evaluation_v2.json")
    secrets = config["secrets"]
    fallback_name = secrets["api_key_env"]
    baseline_key = os.getenv(secrets.get("baseline_api_key_env", "")) or os.getenv(fallback_name)
    router_key = os.getenv(secrets.get("router_api_key_env", "")) or os.getenv(fallback_name)
    if not baseline_key or not router_key:
        raise RuntimeError("OpenRouter API keys are missing; check the .env file for the run's configured key names.")

    base_url = config["provider"]["base_url"]
    baseline_client = OpenRouterClient(baseline_key, base_url, timeout_seconds=30, allow_fallbacks=False)
    router_client = OpenRouterClient(router_key, base_url, timeout_seconds=30, allow_fallbacks=False)
    try:
        records = read_jsonl(run_dir / "generation_records.jsonl")
        semaphore = asyncio.Semaphore(8)
        updated = await asyncio.gather(*[
            fetch_one(item, baseline_client, router_client, semaphore)
            for item in records
        ])
    finally:
        await baseline_client.close()
        await router_client.close()

    rewrite_jsonl(run_dir / "generation_records.jsonl", updated)
    for item in updated:
        raw = item.get("raw_generation_record")
        generation_id = item.get("generation_id")
        if raw and generation_id:
            write_json(run_dir / "raw_generation_records" / f"{generation_id}.json", raw)

    conn_summary = update_connectivity(run_dir, updated)
    paired_summary = update_paired_costs(run_dir, updated)

    summary_path = run_dir / "summary.json"
    overall = read_json(summary_path) if summary_path.exists() else {}
    if conn_summary is not None:
        overall.setdefault("phase_results", {})["connectivity"] = conn_summary
    if paired_summary is not None:
        overall["paired_cost_comparison"] = paired_summary
        overall.setdefault("phase_results", {})["paired_cost_comparison"] = paired_summary
    if overall:
        write_json(summary_path, overall)
        (run_dir / "report.md").write_text(markdown_report(overall.get("run_id", run_dir.name), overall), encoding="utf-8")

    checksums = {}
    for file in sorted(run_dir.rglob("*")):
        if file.is_file() and file.name != "evidence_checksums.json":
            checksums[str(file.relative_to(run_dir))] = {
                "sha256": sha256_file(file),
                "size_bytes": file.stat().st_size,
            }
    write_json(run_dir / "evidence_checksums.json", {
        "run_id": overall.get("run_id", run_dir.name),
        "algorithm": "SHA-256",
        "files": checksums,
        "note": "Hashes detect changes; they are not a digital signature.",
    })

    total = len(updated)
    reconciled = sum(1 for item in updated if item.get("reconciliation_status") == "reconciled")
    pending = sum(1 for item in updated if item.get("generation_id") and item.get("reconciliation_status") != "reconciled")
    return {
        "run_id": overall.get("run_id", run_dir.name),
        "generation_records": total,
        "reconciled": reconciled,
        "pending": pending,
        "connectivity": conn_summary,
        "paired_cost_comparison": paired_summary,
        "evidence_directory": str(run_dir),
        "note": "Metadata GETs only; no model inference calls were made.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile OpenRouter generation records for an existing Evaluation V2 run.")
    parser.add_argument("--run-dir", required=True, help="Existing live_evaluation_v2/<run_id> directory.")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "run_manifest.json").exists():
        parser.error(f"Not an Evaluation V2 run directory: {run_dir}")
    result = asyncio.run(reconcile_run(run_dir))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
