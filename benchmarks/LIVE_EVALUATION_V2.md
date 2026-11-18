# Live End-to-End Evaluation V2

This harness complements `benchmarks/v2_runner.py`. The V2 benchmark directly calls OpenRouter; this live suite sends ModelSync requests through the **running FastAPI server** and compares its cost with a direct single-model baseline.

## Experimental comparison

Both experiments use the exact same 80 prompts from `benchmarks/dataset_v2_80.json`.

- **Baseline:** one direct OpenRouter request to `anthropic/claude-sonnet-5` for each prompt.
- **ModelSync:** one real POST to `/v1/chat/completions` for each prompt. The server calls the classifier and then its selected downstream model.

ModelSync cost is classifier generation cost **plus** downstream generation cost. Baseline cost is the baseline model generation cost. Savings are reported only if all 80 pairs succeeded and all 240 generations have reconciled OpenRouter generation records and actual costs. No catalogue-only estimate is used to claim actual savings. Response-quality scoring is intentionally excluded.

## Required configuration

Use the model IDs in `benchmarks/evaluation_v2.json`. The running FastAPI server uses the application model settings in `.env`.

Recommended environment settings:

```dotenv
OPENROUTER_API_KEY=your_router_key
OPENROUTER_ROUTER_API_KEY=your_router_key
OPENROUTER_BASELINE_API_KEY=your_baseline_key
OPENROUTER_ALLOW_FALLBACKS=false
ROUTER_API_KEY=optional_inbound_router_auth_token

CLASSIFIER_MODEL=z-ai/glm-5.3-flash
CODING_MODEL=deepseek/deepseek-v4.1-flash-1017
TESTING_MODEL=deepseek/deepseek-v4-flash-0731
REASONING_MODEL=openai/gpt-5.6-luna

OPENROUTER_CLASSIFIER_MODEL=z-ai/glm-5.3-flash
OPENROUTER_CODING_MODEL=deepseek/deepseek-v4.1-flash-1017
OPENROUTER_UNIT_TESTING_MODEL=deepseek/deepseek-v4-flash-0731
OPENROUTER_REASONING_MODEL=openai/gpt-5.6-luna
OPENROUTER_BASELINE_MODEL=anthropic/claude-sonnet-5
```

Provider fallbacks are disabled by default for the controlled comparison. The actual provider name returned for each generation is still recorded. If using separate keys, the `OPENROUTER_ROUTER_API_KEY` must be the same key used by the running ModelSync server. The app now prefers that variable over `OPENROUTER_API_KEY`. The harness uses `OPENROUTER_BASELINE_API_KEY` for baseline requests and `OPENROUTER_ROUTER_API_KEY` for generation-record reconciliation. If using one key, set only `OPENROUTER_API_KEY` and leave the two separate variables blank.

Do not commit `.env` or include API keys in logs, screenshots, or reports.

## Run order

### 1. Update your local branch and install dependencies

```powershell
git switch evaluation-v2-preparation
git pull --ff-only origin evaluation-v2-preparation
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

### 2. Run offline validation

```powershell
New-Item -ItemType Directory -Force testing-results/evaluation-v2-preflight | Out-Null
python -m pytest tests/test_evaluation_v2.py tests/test_router.py -v --tb=short --junitxml=testing-results/evaluation-v2-preflight/pytest-junit.xml --log-file=testing-results/evaluation-v2-preflight/pytest.log 2>&1 | Tee-Object -FilePath testing-results/evaluation-v2-preflight/pytest-console.log
if ($LASTEXITCODE -ne 0) { throw 'Offline validation tests failed.' }
```

These tests do not call OpenRouter.

### 3. Run local V2 configuration preflight

```powershell
python -m benchmarks.live_test_suite_v2 --phase preflight
```

This checks dataset size, category distribution, and configured model IDs locally. It makes no provider requests.

### 4. Verify current OpenRouter catalogue and prices

```powershell
python -m benchmarks.v2_runner --catalog-preflight 2>&1 | Tee-Object -FilePath testing-results/evaluation-v2-preflight/catalogue-preflight.log
if ($LASTEXITCODE -ne 0) { throw 'OpenRouter catalogue preflight failed.' }
```

This contacts the public OpenRouter model catalogue but does not request model completions.

### 5. Start the real ModelSync server

Open Terminal 1 at the repository root:

```powershell
python router.py
```

Wait until Uvicorn reports that it is running on `http://127.0.0.1:8000`. Keep this terminal open.

### 6. Run paid live test phases

Open Terminal 2 at the repository root, activate the same virtual environment, and run the phases individually. Each phase creates its own evidence run directory.

Direct connectivity check for all five configured model roles (five short completions):

```powershell
python -m benchmarks.live_test_suite_v2 --phase connectivity --live --i-understand-paid-calls
```

Live FastAPI endpoint checks, one real request per routing category plus health, models, validation and configured inbound-auth checks:

```powershell
python -m benchmarks.live_test_suite_v2 --phase api --live --i-understand-paid-calls
```

**Primary paired cost experiment:** 80 baseline calls + 80 classifier calls + 80 downstream calls. The default run budget stop threshold is USD 5 and applies to the paired experiment. The runner checks after a complete prompt pair, so a final in-flight pair can take the actual spend slightly above the threshold:

```powershell
python -m benchmarks.live_test_suite_v2 --phase paired-cost --live --i-understand-paid-calls --max-cost-usd 5.00
```

Live streaming tests (one prompt per category):

```powershell
python -m benchmarks.live_test_suite_v2 --phase streaming --live --i-understand-paid-calls
```

Live performance tests at concurrency 1, 5 and 10. Default: ten requests at each concurrency level (30 total), so the concurrency-10 scenario actually has ten simultaneous requests. Change the count explicitly with `--performance-requests N`:

```powershell
python -m benchmarks.live_test_suite_v2 --phase performance --live --i-understand-paid-calls --performance-requests 5
```

Reliability checks include configured inbound authentication, an invalid upstream model ID, and controlled local connection failure:

```powershell
python -m benchmarks.live_test_suite_v2 --phase reliability --live --i-understand-paid-calls
```

The live phases make paid inference calls. The `--max-cost-usd` stop threshold currently applies to the 80-prompt paired-cost phase; it is not a provider-enforced hard cap and can be exceeded by the final in-flight prompt pair. The small connectivity/API/streaming/performance phases are separate runs with their own evidence folders and their costs are not included in the paired savings calculation. Do not use the paid commands until catalogue preflight and offline tests pass.

## Evidence folder

Each phase writes a unique directory under `testing-results/live_evaluation_v2/` with:

```text
<run_id>/
├── run.log
├── run_manifest.json
├── inputs/
├── pricing_snapshot.json
├── raw_completions/
├── raw_generation_records/
├── raw_errors/
├── generation_records.jsonl
├── paired_cost_results.json        # paired-cost phase
├── paired_cost_results.csv         # paired-cost phase
├── api_test_results.json/csv       # api phase
├── streaming_results.json/csv      # streaming phase
├── performance_results.json/csv    # performance phase
├── reliability_results.json/csv    # reliability phase
├── summary.json
├── report.md
├── evidence_checksums.json
└── run_failure.json                # if the runner aborts
```

Raw completions preserve the prompts, request parameters, complete response bodies and model outputs. Generation records are reconciled through OpenRouter's generation endpoint. The manifest records the Git commit, branch, software versions, config and dataset hashes, selected model IDs, and the catalogue snapshot timestamp. API-key values are never written.

Checksums can detect later file changes but are not digital signatures. Keep the entire run folder and the OpenRouter Activity export together for the final project evidence.

## Interpretation and limitations

- The paired-cost result compares ModelSync with the specific configured Claude Sonnet 5 baseline, not every possible single-model baseline.
- The paired-cost runner makes the baseline and ModelSync calls for the same input prompt. Their costs are not estimates from the catalogue; each call must have an OpenRouter generation record.
- API, streaming, and performance phase costs are separate operational-test expenditure and are not included in the paired-cost savings percentage.
- The reliability suite labels controlled local faults separately from real upstream failures. It does not deliberately exhaust credits or provoke provider rate limits.
- If the classifier telemetry is absent, malformed, or falls back, the routing result is not counted as a correct prediction.
- No model response-quality scoring is performed in this evaluation.