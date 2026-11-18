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
| CODING | `deepseek/deepseek-v4.1-flash-1017` | $0.036 / $0.50* |
| UNIT_TESTING | `deepseek/deepseek-v4-flash-0731` | $0.0434 / $0.30 |
| REASONING | `openai/gpt-5.6-luna` | $0.20 / $1.20 |
| Baseline | `anthropic/claude-sonnet-5` | $2.00 / $10.00 |

The GLM, DeepSeek V4 Flash, GPT-5.6 Luna, and Claude Sonnet 5 reference prices above were transcribed from user-provided OpenRouter screenshots dated November 12, 2026. The DeepSeek V4.1 Flash 1017 price is from the OpenRouter listing checked November 18, 2026 (*input/output per 1M tokens; verify in preflight). These are retained as references only. Immediately before a run, use the current OpenRouter catalogue snapshot as the recorded catalogue price source, and use reconciled generation records for actual billed cost.

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

## Result artifacts and evidence retention

Each live execution creates a unique directory under `testing-results/evaluation_v2/<run_id>/`:

```text
<run_id>/
├── run.log
├── run_manifest.json
├── inputs/
│   ├── evaluation_v2.json
│   └── dataset_v2_80.json
├── pricing_snapshot.json
├── prompt_results_in_progress.json
├── prompt_results.json
├── prompt_results.csv
├── generation_records.jsonl
├── raw_completions/
├── raw_generation_records/
├── raw_errors/
├── summary.json
├── evidence_checksums.json
└── run_failure.json              # only if the runner aborts
```

- `run.log`: timestamped run events, model calls, outcomes, generation IDs, latency, cost, reconciliation state, and exceptions.
- `run_manifest.json`: run identity, Git commit/branch/working-tree status, Python/platform/package versions, exact model IDs, generation settings, prompt counts, input-file SHA-256 hashes, and pricing-snapshot timestamp. API-key values are never saved.
- `inputs/`: exact copies of the config and 80-prompt dataset used.
- `raw_completions/`: sanitized request body, prompt messages and parameters, complete OpenRouter response including model output, generation ID, and generation metadata. Authorization headers and API keys are excluded.
- `raw_generation_records/`: raw OpenRouter generation metadata keyed by generation ID.
- `raw_errors/`: request messages, status, error payload and exception details for failed calls, without credentials.
- `generation_records.jsonl`: one normalized accounting record per baseline, classifier, or downstream call.
- `prompt_results_in_progress.json`: incremental results after each call, retained if the run is interrupted.
- `prompt_results.csv` and `prompt_results.json`: prompt-level predictions, selected models, token counts, latency, generation IDs, actual costs, catalogue estimates and reconciliation results.
- `pricing_snapshot.json`: time-stamped catalogue entries and raw prices for the five selected model roles.
- `summary.json`: classification/routing metrics, API call counts, reconciliation counts, cost totals, and completion status.
- `evidence_checksums.json`: SHA-256 hashes and file sizes for captured files, to detect later changes.
- `run_failure.json`: exception and progress information if an unexpected failure aborts the runner.

Checksums detect changes after capture; they are not a digital signature or proof that the machine was uncompromised. Preserve the entire run directory, its Git commit, and the OpenRouter Activity export together.

### Capture offline validation evidence

Run the V2 offline tests and retain the console transcript, JUnit report, and pytest log:

```powershell
New-Item -ItemType Directory -Force testing-results/evaluation_v2/preflight | Out-Null
python -m pytest tests/test_evaluation_v2.py -v --tb=short --junitxml testing-results/evaluation_v2/preflight/pytest_junit.xml --log-file testing-results/evaluation_v2/preflight/pytest.log 2>&1 | Tee-Object -FilePath testing-results/evaluation_v2/preflight/pytest_console.log
if ($LASTEXITCODE -ne 0) { throw "Evaluation V2 offline tests failed." }
```

Capture catalogue-preflight output too:

```powershell
python -m benchmarks.v2_runner --catalog-preflight 2>&1 | Tee-Object -FilePath testing-results/evaluation_v2/preflight/catalog_preflight_console.log
if ($LASTEXITCODE -ne 0) { throw "OpenRouter catalogue preflight failed." }
```

These steps do not generate model completions, although catalogue preflight contacts OpenRouter's public model-catalogue endpoint.

### Evidence boundary — what this runner proves

The V2 runner directly calls OpenRouter to compare a fixed single-model baseline against classifier-plus-downstream routing. It is designed to preserve live outputs, per-call usage/cost evidence, and reproducible run metadata.

It does **not** send requests through the running ModelSync FastAPI server. By itself it does not prove live `/v1/chat/completions` end-to-end behaviour, inbound API authentication, streaming/SSE, concurrent load, or API-level failure handling. Those need a separate live-server integration, performance, and reliability harness. The legacy `run_evaluation_suite.py` uses a mock provider for its API endpoint and failure-injection stages, and its performance section is primarily offline controlled mock; do not present those stages as paid-provider live results. The V2 offline pytest suite also has to be executed locally before claiming it passed.

## Cross-check in OpenRouter Activity

Use separate baseline and router API keys, filter Activity by the corresponding keys and test time range, export the usage data, and compare its totals with the per-generation records. OpenRouter describes Activity and its Analytics API as providing spend and request breakdowns by API key, model, provider, and generation: https://openrouter.ai/blog/announcements/activity-dashboard/.

Keep inference expenditure separate from fees paid when purchasing credits.

### Why DeepSeek V4.1 Flash 1017 is the coding candidate

The coding route uses DeepSeek V4.1 Flash 1017 rather than GLM 5.3 Flash because it placed strongly in the published DeepSWE v1.1 comparison (77.0% in the cited comparison) and is listed among OpenRouter's leading coding models by real usage. This is a selection hypothesis, not a claim that it has been proven superior on the ModelSync workload. Evaluation V2 should retain both model IDs in the candidate comparison and test the same 27 CODING prompts before the final black-book claim. GLM 5.3 Flash remains the classifier model.
