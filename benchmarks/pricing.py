"""Authoritative pricing models and cost calculators for benchmark evaluation.

Data Source: OpenRouter Public API (https://openrouter.ai/api/v1/models)
Date Accessed: 2026-10-24
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ModelPrice:
    model_id: str
    prompt_price_per_million: float
    completion_price_per_million: float
    is_free: bool = False
    source: str = "OpenRouter Catalog (2026-10-24)"
    notes: str = ""

    @property
    def prompt_cost_per_token(self) -> float:
        return self.prompt_price_per_million / 1_000_000.0

    @property
    def completion_cost_per_token(self) -> float:
        return self.completion_price_per_million / 1_000_000.0


# Authoritative published pricing catalog as verified on 2026-10-24
PRICE_CATALOG: Dict[str, ModelPrice] = {
    # Flagship Reasoning / Default Baseline
    "nvidia/nemotron-3-ultra-550b-a55b": ModelPrice(
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        prompt_price_per_million=0.60,
        completion_price_per_million=2.40,
        is_free=False,
        notes="Flagship 550B reasoning model; primary baseline candidate",
    ),
    "nvidia/nemotron-3-ultra-550b-a55b:free": ModelPrice(
        model_id="nvidia/nemotron-3-ultra-550b-a55b:free",
        prompt_price_per_million=0.00,
        completion_price_per_million=0.00,
        is_free=True,
        notes="Promotional free endpoint for Nemotron Ultra",
    ),
    # Coding Model
    "qwen/qwen3.8-27b": ModelPrice(
        model_id="qwen/qwen3.8-27b",
        prompt_price_per_million=0.20,
        completion_price_per_million=2.50,
        is_free=False,
        notes="Specialized 27B coding/general model",
    ),
    "qwen/qwen3.8-27b:free": ModelPrice(
        model_id="qwen/qwen3.8-27b:free",
        prompt_price_per_million=0.00,
        completion_price_per_million=0.00,
        is_free=True,
        notes="Deprecated free slug; paid version slug is qwen/qwen3.8-27b",
    ),
    # Unit Testing Model
    "google/gemma-4-26b-a4b-it": ModelPrice(
        model_id="google/gemma-4-26b-a4b-it",
        prompt_price_per_million=0.0675,
        completion_price_per_million=0.225,
        is_free=False,
        notes="Cost-efficient 26B instruction model; testing target",
    ),
    "google/gemma-4-26b-a4b-it:free": ModelPrice(
        model_id="google/gemma-4-26b-a4b-it:free",
        prompt_price_per_million=0.00,
        completion_price_per_million=0.00,
        is_free=True,
        notes="Promotional free endpoint for Gemma 26b (subject to shared rate limits)",
    ),
    # Alternative Models
    "google/gemma-4-31b-it": ModelPrice(
        model_id="google/gemma-4-31b-it",
        prompt_price_per_million=0.20,
        completion_price_per_million=0.40,
        is_free=False,
        notes="Gemma 31b instruction model",
    ),
    "google/gemma-4-31b-it:free": ModelPrice(
        model_id="google/gemma-4-31b-it:free",
        prompt_price_per_million=0.00,
        completion_price_per_million=0.00,
        is_free=True,
        notes="Gemma 31b free endpoint",
    ),
    "cohere/north-mini-code:free": ModelPrice(
        model_id="cohere/north-mini-code:free",
        prompt_price_per_million=0.00,
        completion_price_per_million=0.00,
        is_free=True,
        notes="Cohere code free endpoint",
    ),
    # Classifier Model
    "poolside/laguna-s-2.1:free": ModelPrice(
        model_id="poolside/laguna-s-2.1:free",
        prompt_price_per_million=0.00,
        completion_price_per_million=0.00,
        is_free=True,
        notes="Lightweight classifier endpoint",
    ),
    # Standard Reference Baselines for Cross-Comparison
    "openai/gpt-4o-2024-11-20": ModelPrice(
        model_id="openai/gpt-4o-2024-11-20",
        prompt_price_per_million=2.50,
        completion_price_per_million=10.00,
        is_free=False,
        notes="OpenAI GPT-4o reference baseline",
    ),
    "openai/gpt-4o-mini": ModelPrice(
        model_id="openai/gpt-4o-mini",
        prompt_price_per_million=0.15,
        completion_price_per_million=0.60,
        is_free=False,
        notes="OpenAI GPT-4o-mini reference economy model",
    ),
}

# Mapping from free-tier aliases to their commercial catalog equivalent
PAID_COUNTERPART_MAPPING: Dict[str, str] = {
    "nvidia/nemotron-3-ultra-550b-a55b:free": "nvidia/nemotron-3-ultra-550b-a55b",
    "qwen/qwen3.8-27b:free": "qwen/qwen3.8-27b",
    "google/gemma-4-26b-a4b-it:free": "google/gemma-4-26b-a4b-it",
    "google/gemma-4-31b-it:free": "google/gemma-4-31b-it",
    "poolside/laguna-s-2.1:free": "poolside/laguna-s-2.1:free",
}


def lookup_model_price(model_id: str, use_paid_counterpart: bool = False) -> Optional[ModelPrice]:
    """Look up model pricing, optionally resolving free-tier endpoints to paid equivalents."""
    target_id = model_id
    if use_paid_counterpart and target_id in PAID_COUNTERPART_MAPPING:
        target_id = PAID_COUNTERPART_MAPPING[target_id]

    if target_id in PRICE_CATALOG:
        return PRICE_CATALOG[target_id]
    
    # Strip :free suffix as fallback
    base_id = target_id.replace(":free", "")
    if base_id in PRICE_CATALOG:
        return PRICE_CATALOG[base_id]
    return None


def calculate_cost(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    use_paid_counterpart: bool = False,
) -> Tuple[float, str, Optional[ModelPrice]]:
    """Calculate the cost in USD for a given token usage.
    
    Formula:
      cost = (prompt_tokens * prompt_price_per_M + completion_tokens * completion_price_per_M) / 1,000,000
      
    Returns:
      (cost_usd, status_label, price_object)
    """
    price = lookup_model_price(model_id, use_paid_counterpart=use_paid_counterpart)
    if not price:
        return 0.0, "PRICE_UNAVAILABLE", None

    cost = (
        prompt_tokens * price.prompt_price_per_million
        + completion_tokens * price.completion_price_per_million
    ) / 1_000_000.0

    if price.is_free and not use_paid_counterpart:
        return 0.0, "MEASURED_FREE", price

    status = "SIMULATED_PAID_CATALOG" if use_paid_counterpart else "CALCULATED_CATALOG"
    return cost, status, price


def calculate_savings_percentage(baseline_cost: float, router_cost: float) -> Optional[float]:
    """Calculate cost savings percentage:
    
    savings (%) = ((baseline_total_cost - router_total_cost) / baseline_total_cost) * 100
    
    Returns None if baseline_cost <= 0 to avoid zero-division or misleading infinite savings.
    """
    if baseline_cost <= 0.0:
        return None
    return ((baseline_cost - router_cost) / baseline_cost) * 100.0
