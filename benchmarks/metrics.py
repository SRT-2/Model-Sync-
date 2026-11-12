"""Evaluation metrics for LLM routing accuracy, latency, and cost efficiency."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


CATEGORIES = ["CODING", "UNIT_TESTING", "REASONING"]


def compute_confusion_matrix(
    actual: List[str], predicted: List[str], categories: Optional[List[str]] = None
) -> Dict[str, Dict[str, int]]:
    """Generate a confusion matrix dict: {actual_class: {predicted_class: count}}."""
    cats = categories or CATEGORIES
    matrix: Dict[str, Dict[str, int]] = {c: {p: 0 for p in cats} for c in cats}
    for act, pred in zip(actual, predicted):
        act_key = act if act in matrix else "OTHER"
        pred_key = pred if pred in (cats) else "OTHER"
        if act_key not in matrix:
            matrix[act_key] = {p: 0 for p in cats}
        if pred_key not in matrix[act_key]:
            matrix[act_key][pred_key] = 0
        matrix[act_key][pred_key] += 1
    return matrix


def compute_classification_metrics(
    actual: List[str], predicted: List[str], categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute overall accuracy and per-category precision, recall, and F1 score."""
    cats = categories or CATEGORIES
    total = len(actual)
    if total == 0:
        return {"accuracy": 0.0, "total": 0, "categories": {}}

    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    overall_accuracy = correct / total

    category_metrics: Dict[str, Dict[str, float]] = {}
    for cat in cats:
        tp = sum(1 for a, p in zip(actual, predicted) if a == cat and p == cat)
        fp = sum(1 for a, p in zip(actual, predicted) if a != cat and p == cat)
        fn = sum(1 for a, p in zip(actual, predicted) if a == cat and p != cat)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        category_metrics[cat] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    # Macro averages
    macro_p = sum(m["precision"] for m in category_metrics.values()) / len(category_metrics)
    macro_r = sum(m["recall"] for m in category_metrics.values()) / len(category_metrics)
    macro_f1 = sum(m["f1"] for m in category_metrics.values()) / len(category_metrics)

    return {
        "accuracy": round(overall_accuracy, 4),
        "correct": correct,
        "total": total,
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "categories": category_metrics,
        "confusion_matrix": compute_confusion_matrix(actual, predicted, cats),
    }


def compute_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Compute mean, median (P50), P90, P95, std dev, min, and max latency."""
    if not latencies:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    s = sorted(latencies)
    n = len(s)

    def percentile(p: float) -> float:
        if n == 1:
            return s[0]
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s[int(k)]
        d0 = s[int(f)] * (c - k)
        d1 = s[int(c)] * (k - f)
        return d0 + d1

    mean_val = sum(s) / n
    variance = sum((x - mean_val) ** 2 for x in s) / n if n > 1 else 0.0
    std_val = math.sqrt(variance)

    return {
        "count": n,
        "mean": round(mean_val, 4),
        "median": round(percentile(50), 4),
        "p90": round(percentile(90), 4),
        "p95": round(percentile(95), 4),
        "std": round(std_val, 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
    }
