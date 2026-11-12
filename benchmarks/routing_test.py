"""Routing correctness benchmark: evaluates the live classifier and model selection."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.providers.together import TogetherProvider
from app.api.routes import select_target_model
from benchmarks.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("routing_test")


async def run_routing_evaluation(dataset_path: Path, live: bool = True) -> dict:
    settings = get_settings()
    with open(dataset_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    provider = TogetherProvider(settings) if live else None
    
    actual_categories = []
    predicted_categories = []
    model_selections = []
    latencies = []
    raw_responses = []

    logger.info("Evaluating routing correctness on %d prompts using classifier: %s (live=%s)",
                len(prompts), settings.classifier_model, live)

    for idx, p in enumerate(prompts):
        prompt_id = p["id"]
        expected_cat = p["category"]
        text = p["prompt"]
        actual_categories.append(expected_cat)

        t0 = time.perf_counter()
        if live:
            try:
                pred = await provider.classify(settings.classifier_model, text)
            except Exception as e:
                logger.error("Classifier error on %s: %s", prompt_id, e)
                pred = "CODING"
        else:
            pred = expected_cat
        lat = time.perf_counter() - t0
        latencies.append(lat)
        predicted_categories.append(pred)

        selected_model = select_target_model(pred, settings)
        expected_model = select_target_model(expected_cat, settings)
        model_correct = (selected_model == expected_model)
        model_selections.append({
            "prompt_id": prompt_id,
            "expected_category": expected_cat,
            "predicted_category": pred,
            "selected_model": selected_model,
            "expected_model": expected_model,
            "model_correct": model_correct,
            "latency_s": round(lat, 4),
        })
        logger.info("[%d/%d] %s: Expected=%s, Predicted=%s, Correct=%s (%.2fs)",
                    idx + 1, len(prompts), prompt_id, expected_cat, pred, (expected_cat == pred), lat)
        if live:
            await asyncio.sleep(0.5)  # throttle

    metrics = compute_classification_metrics(actual_categories, predicted_categories)
    model_correct_count = sum(1 for m in model_selections if m["model_correct"])
    model_accuracy = model_correct_count / len(model_selections) if model_selections else 0.0

    result = {
        "classifier_model": settings.classifier_model,
        "total_prompts": len(prompts),
        "classification_accuracy": metrics["accuracy"],
        "model_selection_correctness": round(model_accuracy, 4),
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "category_metrics": metrics["categories"],
        "mean_latency_s": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        "details": model_selections,
    }

    out_file = PROJECT_ROOT / "benchmarks" / "results" / "routing_correctness.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved routing correctness results to %s", out_file)
    return result


if __name__ == "__main__":
    asyncio.run(run_routing_evaluation(PROJECT_ROOT / "benchmarks" / "dataset.json", live=True))
