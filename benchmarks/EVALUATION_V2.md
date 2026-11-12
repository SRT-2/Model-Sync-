# Evaluation V2 — OpenRouter Live Testing

This is a separate evaluation path. It preserves the original benchmark, dataset, results, and runner. The V2 dataset contains 80 unique prompts and is used for live classification/routing and baseline-versus-ModelSync inference-cost comparison.

## Scope

For each prompt, the runner makes three paid inference calls:
1. Baseline: send the original user prompt to one fixed baseline model.
2. Classifier: send the classifier system prompt and user prompt to the configured classifier model.
3. Downstream: send the original user prompt to the model selected for the classifier category.

ModelSync cost includes both classifier and downstream calls. Response-quality scoring is intentionally excluded.

This benchmark runner exercises the real classifier + model-selection + downstream inference path directly through OpenRouter. It does not replace the separate FastAPI end-to-end, streaming, performance, and failure-injection suites.

## Dataset

File: `benchmarks/dataset_v2_80.json`

| Category | Prompts |
|---|---:|
| CODING | 27 |
| UNIT_TESTING | 27 |
| REASONING | 26 |
| **Total** | **80** |

Each item has `id`, `category`, `difficulty` (`clear`, `boundary`, `complex`), `prompt`, and `description`. There are no reference answers or rubric fields because this experiment does not grade response quality.

## Configure exact paid model IDs

The current selected model mapping is recorded in `benchmarks/evaluation_v2.json`:

| Role | OpenRouter model ID | Screenshot reference input / output per 1M tokens |
|---|---|---:|
| Classifier | `z-ai/glm-5.3-flash` | $0.0399 / $0.0784 |
| CODING | `z-ai/glm-5.3-flash` | $0.0399 / $0.0784 |
| UNIT_TESTING | `deepseek/deepseek-v4-flash-0731` | $0.0434 / $0.30 |
| REASONING | `openai/gpt-5.6-luna` | $0.20 / $1.20 |
| Baseline | `anthropic/claude-sonnet-5` | $2.00 / $10.00 |

The reference prices above were transcribed from user-provided OpenRouter screenshots dated November 12, 2026. They are retained as a reference only. Immediately before a run, use the current OpenRouter catalogue snapshot as the recorded catalogue price source, and use reconciled generation records for actual billed cost.

V2 rejects unknown, unpriced, or zero-priced model entries, because this is a paid-model experiment. Model IDs can be overridden with environment variables if a model is replaced.

Set these in `.env`:

```dotenv
OPENROUTER_API_KEY=...
OPENROUTER_BASELINE_API_KEY=...
OPENROUTER_ROUTER_API_KEY=...
OPENROUTER_BASELINE_MODEL=author/model-id
OPENROUTER_CLASSIFIER_MODEL=author/model-id
OPENROUTER_CODING_MODEL=author/model-id
OPENROUTER_UNIT_TESTING_MODEL=author/model-id
OPENROUTER_REASONING_MODEL=author/model-id
```

Separate baseline and router keys are recommended so OpenRouter Activity can separate the two experiments. If the separate keys are omitted, the runner uses `OPENROUTER_API_KEY` for both. Do not commit `.env` or API keys.

## Execution sequence

1. Run offline validation tests:

```powershell
python -m pytest tests/test_evaluation_v2.py -q
```

2. After reviewing the selected model IDs, capture the public model catalogue and execution-time pricing snapshot. This does not make inference calls. Confirm the catalogue includes every exact ID and compare its prices with the screenshot reference stored in the config:

```powershell
python -m benchmarks.v2_runner --catalog-preflight
```

3. Execute paid inference only after reviewing the model IDs and price snapshot:

```powershell
python -m benchmarks.v2_runner --live --i-understand-paid-calls --max-cost-usd 1.00
```

Both `--live` and `--i-understand-paid-calls` are required. The default cost guard is configured in `benchmarks/evaluation_v2.json`; override it with `--max-cost-usd`.

The budget guard checks cost after each generation. When actual cost is not yet available, a buffered catalogue estimate is used only for the budget guard, never reported as actual billing. If a successful generation has neither actual cost nor a catalogue estimate, the run stops.

## Usage and generation reconciliation

Every non-streaming Chat Completions call requests usage metadata with `usage: { include: true }`. The runner records the completion response `id` (the OpenRouter generation ID), model, usage, latency, and provider information where returned.

After each successful completion, it calls:

```http
GET https://openrouter.ai/api/v1/generation?id=<generation-id>
```

The returned metadata record is saved under `raw_generation_records/`. Its `data.total_cost` (or `data.cost` if returned) is preferred over the completion response `usage.cost`. If none is available, actual cost stays unknown; it is never silently set to zero.

The runner retains response token counts as well as generation metadata. OpenRouter documentation notes that the response exposes token usage and cost and that the generation endpoint can be queried using the completion ID for request statistics and cost. See https://openrouter.ai/docs/api/api-reference/generations/get-generation and https://openrouter.ai/docs/api-reference/overview.

The catalogue snapshot records the exact model IDs and raw `pricing` fields at run time. Model prices in the catalogue are per token. Catalogue estimates are stored separately from actual billed cost and are not substituted into savings calculations.

## Result artifacts

Each execution creates a unique directory under `testing-results/evaluation_v2/<run_id>/`:

```text
pricing_snapshot.json
prompt_results.json
prompt_results.csv
generation_records.jsonl
raw_generation_records/
summary.json
```

- `prompt_results.csv`: per-prompt models, token counts, generation IDs, actual costs, catalogue estimates, and reconciliation status.
- `generation_records.jsonl`: one accounting record per baseline, classifier, or downstream call.
- `raw_generation_records/`: raw OpenRouter generation metadata keyed by generation ID.
- `pricing_snapshot.json`: timestamped catalogue entries and raw prices.
- `summary.json`: classification/routing metrics, API call counts, reconciliation counts, cost totals, and completion status.

An actual savings percentage is emitted only when all 80 paired prompt comparisons are complete and an actual cost is available for every baseline, classifier, and downstream call. Partial runs or missing cost records produce no savings percentage.

## Cross-check in OpenRouter Activity

Use separate baseline and router API keys, filter Activity by the corresponding keys and test time range, export the usage data, and compare its totals with the per-generation records. OpenRouter describes Activity and its Analytics API as providing spend and request breakdowns by API key, model, provider, and generation: https://openrouter.ai/blog/announcements/activity-dashboard/.

Keep inference expenditure separate from fees paid when purchasing credits.