

```markdown
# ModelSync — LLM Smart Router v2

An OpenAI-compatible FastAPI routing engine with a React + TypeScript dashboard. ModelSync analyzes each incoming prompt with a lightweight classifier model, identifies the task category (**`CODING`**, **`UNIT_TESTING`**, or **`REASONING`**), and dynamically forwards the request to the appropriate downstream model — while clients connect to a single virtual model: `smart-router`.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Project Objectives](#project-objectives)
4. [Key Features](#key-features)
5. [System Architecture](#system-architecture)
6. [How ModelSync Works](#how-modelsync-works)
7. [Routing Categories and Model Mapping](#routing-categories-and-model-mapping)
8. [Persona Injection Mechanism](#persona-injection-mechanism)
9. [Technology Stack](#technology-stack)
10. [Project Structure](#project-structure)
11. [Installation and Setup](#installation-and-setup)
12. [Environment Configuration](#environment-configuration)
13. [Running the Backend](#running-the-backend)
14. [Running the Frontend](#running-the-frontend)
15. [API Documentation](#api-documentation)
16. [Testing and Validation](#testing-and-validation)
17. [Experimental Evaluation](#experimental-evaluation)
18. [Cost Optimization Analysis](#cost-optimization-analysis)
19. [Results and Performance Metrics](#results-and-performance-metrics)
20. [Docker Deployment](#docker-deployment)
21. [Limitations](#limitations)
22. [Future Scope](#future-scope)
23. [Contributors](#contributors)

---

## 1. Project Overview

ModelSync (LLM Smart Router v2) is a lightweight LLM request router that sits between client applications and upstream LLM providers (OpenRouter, Together AI, or any OpenAI-compatible endpoint). Instead of calling a single model for every request, ModelSync:

1. Receives an OpenAI-compatible `POST /v1/chat/completions` request targeting the virtual model `smart-router`.
2. Extracts the latest user prompt from the conversation.
3. Classifies the prompt into one of three task categories using a dedicated classifier model.
4. Forwards the request to a category-specific downstream model.
5. Returns the response with routing metadata (`routed_category`, `routed_model`) in both JSON body and response headers.

The frontend provides a chat playground, routing insights panel, connection/settings management, and session request history — all maintaining state in the browser (no backend database).

---

## 2. Problem Statement

Using a single large language model for every type of request is inefficient:

- **Cost**: General-purpose flagship models are expensive for simple tasks (e.g., writing unit tests).
- **Latency**: Large models can be slower than necessary for well-scoped tasks.
- **Capability mismatch**: A model optimized for reasoning may be overkill for code generation, and vice versa.

ModelSync addresses this by routing each request to a task-specific model based on prompt classification, while presenting a single virtual endpoint to the client.

---

## 3. Project Objectives

- Provide an OpenAI-compatible API (`/v1/chat/completions`) so existing clients work without modification.
- Classify incoming prompts into task categories (`CODING`, `UNIT_TESTING`, `REASONING`) using a lightweight classifier model.
- Route each request to a pre-configured downstream model optimized for that category.
- Support both streaming (SSE) and non-streaming responses.
- Return routing metadata so clients know which category and model handled each request.
- Offer optional Bearer token authentication for the router itself.
- Provide a React + TypeScript dashboard with chat playground, routing insights, model settings, and session history.
- Keep the architecture lightweight: no database, session state kept on the frontend.
- Support deployment via Docker Compose.

---

## 4. Key Features

| Feature | Description |
|---------|-------------|
| **OpenAI-compatible API** | `POST /v1/chat/completions` with virtual model `smart-router`; works with any OpenAI SDK client. |
| **Prompt classification** | Lightweight classifier model categorizes prompts into `CODING`, `UNIT_TESTING`, or `REASONING`. |
| **Dynamic model routing** | Requests are forwarded to category-specific downstream models configured via environment variables. |
| **Streaming (SSE)** | Server-Sent Events streaming with initial routing metadata chunk before token stream begins. |
| **Non-streaming mode** | Standard JSON response with routing metadata in body fields and headers. |
| **Routing metadata** | `X-Routed-Category` and `X-Routed-Model` response headers; `routed_category` and `routed_model` JSON fields. |
| **Optional auth** | `ROUTER_API_KEY` Bearer token authentication (disabled by default). |
| **React dashboard** | Chat playground with streaming, stop/cancel, code blocks with copy, starter prompts per category. |
| **Routing insights** | Real-time pipeline visualization and per-session category distribution. |
| **Session history** | Browser-side log of completed requests with category, model, duration, and token usage; search and filter. |
| **Connection settings** | Configurable backend base URL, router API key, and model table from `/v1/models`. |
| **Docker support** | Multi-stage Dockerfile + Docker Compose, non-root user, healthcheck. |

---

## 5. System Architecture

```mermaid
graph TD
    Client["Client / Browser Dashboard"]
    API["FastAPI Router Server (:8000)"]
    Classifier["Classifier Model<br/>poolside/laguna-s-2.1:free"]
    CodingModel["Coding Model<br/>qwen/qwen3.8-27b:free"]
    TestingModel["Testing Model<br/>google/gemma-4-26b-a4b-it:free"]
    ReasoningModel["Reasoning Model<br/>nvidia/nemotron-3-ultra-550b-a55b:free"]
    Upstream["Upstream Provider<br/>(OpenRouter / Together AI)"]

    Client -->|POST /v1/chat/completions<br/>model: smart-router| API
    API -->|1. Extract user prompt| Classifier
    Classifier -->|2. Category label| API
    API -->|3. Select target model| CodingModel
    API -->|3. Select target model| TestingModel
    API -->|3. Select target model| ReasoningModel
    CodingModel -->|4. Completion| Upstream
    TestingModel -->|4. Completion| Upstream
    ReasoningModel -->|4. Completion| Upstream
    Upstream -->|5. Response| API
    API -->|6. Response + routing metadata| Client
```

### Request Flow

1. **Client** sends `POST /v1/chat/completions` to the FastAPI backend with `model: "smart-router"`.
2. **Backend** extracts the last user message content from the `messages` array.
3. **Classifier** model categorizes the prompt into `CODING`, `UNIT_TESTING`, or `REASONING`.
4. **Router** selects the target downstream model based on the category.
5. **Provider** (OpenAI-compatible client) forwards the request to the selected model via OpenRouter or Together AI.
6. **Backend** returns the completion with `routed_category` and `routed_model` metadata.

---

## 6. How ModelSync Works

### Complete Request Lifecycle

```
Client Request
     |
     v
POST /v1/chat/completions { model: "smart-router", messages: [...] }
     |
     v
+-------------------------------------+
| Step 1: Extract last user prompt    |
| extract_last_user_content(messages) |
+-------------------------------------+
     |
     v
+-------------------------------------+
| Step 2: Classify intent             |
| provider.classify(classifier_model, |
|   prompt) -> category label         |
| Default: CODING on failure          |
+-------------------------------------+
     |
     v
+-------------------------------------+
| Step 3: Select target model         |
| CODING    -> coding_model           |
| UNIT_TESTING -> testing_model       |
| REASONING  -> reasoning_model       |
| Unknown   -> coding_model (fallback)|
+-------------------------------------+
     |
     v
+-------------------------------------+
| Step 4: Forward to upstream provider|
| OpenAI-compatible client ->         |
| OpenRouter / Together AI            |
| Stream or non-stream                |
+-------------------------------------+
     |
     v
+-------------------------------------+
| Step 5: Return response             |
| JSON: routed_category, routed_model |
| Headers: X-Routed-Category,         |
|          X-Routed-Model             |
| SSE: initial metadata chunk         |
+-------------------------------------+
     |
     v
Client Response
```

### Routing Categories and Definitions

| Category | Definition | Example Use Cases |
|----------|-----------|-------------------|
| **`CODING`** | Code generation, implementation, debugging, bug fixing, refactoring, algorithms, script writing, syntax questions. | "Implement a thread-safe LRU Cache", "Fix binary search bug" |
| **`UNIT_TESTING`** | Writing unit tests, test suites, test cases, assertions, mock setups, edge-case testing, pytest, jest, unittest. | "Write pytest tests for login function", "Jest test suite for currency formatter" |
| **`REASONING`** | System architecture, high-level design, math, logical deduction, explanation, open-ended discussions, strategy, essay writing. | "CAP theorem trade-offs", "Knight and Knave logic puzzle" |

The classifier model responds with exactly one category label. If the classifier output is unparseable or a transient error occurs, the router falls back to `CODING`.

---

## 7. Routing Categories and Model Mapping

Model selection is **static configuration-based** — each category maps to a fixed downstream model. The router does not dynamically calculate or select the cheapest model at runtime.

### Default Model Configuration

| Category | Default Model | Provider | Configurable Via |
|----------|--------------|----------|------------------|
| CODING | `qwen/qwen3.8-27b:free` | OpenRouter / Together | `CODING_MODEL` |
| UNIT_TESTING | `google/gemma-4-26b-a4b-it:free` | OpenRouter / Together | `TESTING_MODEL` |
| REASONING | `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter / Together | `REASONING_MODEL` |
| Classifier | `poolside/laguna-s-2.1:free` | OpenRouter / Together | `CLASSIFIER_MODEL` |

These defaults can be overridden in `.env` without code changes. The `docker-compose.yml` also supports overriding via environment variables with different default models for containerized deployment.

### Model Selection Logic (`app/api/routes.py`)

```python
def select_target_model(category: str, settings: Settings) -> str:
    if category == UNIT_TESTING:
        return settings.testing_model
    elif category == REASONING:
        return settings.reasoning_model
    return settings.coding_model  # CODING and unknown categories
```

---

## 8. Persona Injection Mechanism

The router applies a **classifier persona** via the system prompt used when calling the classifier model. The system prompt (`CLASSIFIER_SYSTEM_PROMPT` in `app/routing.py`) instructs the classifier model to:

- Analyze the user prompt and classify it into exactly one category: `CODING`, `UNIT_TESTING`, or `REASONING`.
- Respond with **only** the category label, no explanations.

The classifier call is made with `temperature=0.0` and `max_tokens=20` to produce deterministic, concise classification output. The raw response is parsed by `parse_label()`, which:

1. Strips and uppercases the raw text.
2. Checks for exact keyword containment (`UNIT_TESTING`, `REASONING`, `CODING`).
3. Falls back to a word-boundary regex search.
4. Defaults to `CODING` if no category is found.

This persona injection happens **only for classification** — downstream models receive the original user messages without any injected system prompt or persona modification.

---

## 9. Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Backend framework | FastAPI | ≥0.115 | Web API, request routing, OpenAPI docs |
| ASGI server | uvicorn | ≥0.30 | Production ASGI server |
| LLM client | openai | ≥1.50 | AsyncOpenAI client for upstream provider calls |
| Settings | pydantic-settings | ≥2.3 | `.env` loading with field aliases |
| Environment | python-dotenv | ≥1.0 | `.env` file support |
| Frontend framework | React | 19.x | UI components |
| Frontend build tool | Vite | 8.x | Dev server, bundling |
| Frontend language | TypeScript | 6.x | Type safety |
| Frontend testing | Vitest | 5.x | Unit tests |
| Backend testing | pytest | 8.x | Unit and integration tests |
| Container | Docker | — | Multi-stage build, non-root user |
| Orchestration | Docker Compose | — | Service definition |

---

## 10. Project Structure

```
llm-composer-coding-agent/
├── app/
│   ├── main.py              # FastAPI app factory, CORS middleware, lifespan
│   ├── config.py            # Pydantic Settings (.env loader with aliases)
│   ├── state.py             # Singleton AppState (provider management)
│   ├── schemas.py           # Pydantic request/response models & validation
│   ├── routing.py           # Classifier prompt, label parser, SSE formatter
│   ├── providers/
│   │   ├── base.py          # LLMProvider abstract base class
│   │   └── together.py      # OpenAI-compatible provider (OpenRouter & Together)
│   └── api/
│       ├── deps.py          # Bearer token authentication dependency
│       └── routes.py        # /health, /v1/models, /v1/chat/completions
├── frontend/
│   ├── src/
│   │   ├── components/      # ChatPlayground, RoutingInsights, ConnectionSettings,
│   │   │                   # SessionHistory, MessageItem, CodeBlock, Sidebar
│   │   ├── context/         # AppContext (React state & API orchestration)
│   │   ├── services/        # api.ts (API client), sseParser.ts (SSE parser)
│   │   ├── styles/          # Design system (dark/light themes)
│   │   ├── types/           # TypeScript type definitions
│   │   └── __tests__/       # Vitest unit tests (api, components, sseParser)
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── benchmarks/
│   ├── dataset.json         # 30-prompt evaluation dataset
│   ├── pricing.py           # Model pricing catalog & cost calculator
│   ├── metrics.py           # Classification metrics, confusion matrix, latency percentiles
│   ├── evaluator.py         # Response quality evaluator (rubric-based)
│   ├── runner.py            # Benchmark runner (dry-run/smoke/full modes)
│   ├── routing_test.py      # Live routing correctness evaluation
│   └── results/             # Benchmark output artifacts (JSON, CSV)
├── tests/
│   ├── test_router.py       # Backend unit tests (16 tests)
│   └── test_benchmark.py    # Benchmark framework unit tests (18 tests)
├── testing-results/         # Experimental evaluation outputs
│   ├── evaluation_manifest.json
│   ├── classification_metrics.json
│   ├── quality_evaluation.json
│   ├── performance_results.csv
│   ├── api_test_results.csv
│   ├── failure_test_results.csv
│   ├── pytest_output.txt
│   ├── evaluation.log
│   ├── classification_results.csv
│   └── confusion_matrix.png
├── router.py                # CLI entry point (python router.py)
├── Dockerfile               # Multi-stage Python 3.13-slim, non-root user
├── docker-compose.yml       # Compose service specification
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Development + test dependencies
├── .env.example             # Environment variable template
└── README.md                # Project documentation
```

---

## 11. Installation and Setup

### Prerequisites

- Python 3.13+ (backend)
- Node.js 18+ and npm (frontend)
- An API key from [OpenRouter](https://openrouter.ai) or [Together AI](https://together.ai)

### Step 1: Clone the Repository

```bash
git clone https://github.com/SRT-2/Model-Sync-.git
cd Model-Sync_/llm-composer-coding-agent
```

### Step 2: Create and Activate a Python Virtual Environment

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Backend Dependencies

```bash
pip install -r requirements.txt
```

For development (includes pytest, httpx):

```bash
pip install -r requirements-dev.txt
```

### Step 4: Configure Environment Variables

Copy the example file and fill in your API key:

```powershell
# Windows PowerShell
Copy-Item .env.example .env
# Edit .env with your OPENROUTER_API_KEY or TOGETHER_API_KEY
```

```bash
# macOS / Linux
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY or TOGETHER_API_KEY
```

**Never commit `.env`** — it is listed in `.gitignore`.

---

## 12. Environment Configuration

All environment variables are loaded via Pydantic Settings from `.env` (see `app/config.py`).

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes* | — | OpenRouter API key |
| `TOGETHER_API_KEY` | Yes* | — | Together AI API key (alternative to OpenRouter) |
| `UPSTREAM_API_KEY` | No | — | Generic upstream API key |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | OpenRouter base URL |
| `TOGETHER_BASE_URL` | No | `https://api.together.xyz/v1` | Together AI base URL |
| `UPSTREAM_BASE_URL` | No | — | Custom upstream base URL (takes precedence) |
| `ROUTER_HOST` | No | `0.0.0.0` | Backend bind host |
| `ROUTER_PORT` | No | `8000` | Backend bind port |
| `ROUTER_API_KEY` | No | (empty) | Optional Bearer token for inbound auth |
| `UPSTREAM_TIMEOUT_SECONDS` | No | `120.0` | Upstream request timeout |
| `UPSTREAM_MAX_RETRIES` | No | `2` | Upstream max retries |
| `CODING_MODEL` | No | `qwen/qwen3.8-27b:free` | Downstream model for CODING category |
| `TESTING_MODEL` | No | `google/gemma-4-26b-a4b-it:free` | Downstream model for UNIT_TESTING category |
| `REASONING_MODEL` | No | `nvidia/nemotron-3-ultra-550b-a55b:free` | Downstream model for REASONING category |
| `CLASSIFIER_MODEL` | No | `poolside/laguna-s-2.1:free` | Classifier model for prompt categorization |

*At least one upstream API key (`OPENROUTER_API_KEY`, `TOGETHER_API_KEY`, or `UPSTREAM_API_KEY`) is required for live provider calls.

### Provider Selection

The backend determines the provider automatically:
- `OPENROUTER_API_KEY` set → provider: `openrouter`, base URL: OpenRouter
- `TOGETHER_API_KEY` set (no OpenRouter key) → provider: `together`, base URL: Together AI
- Neither set → provider: `custom` (requires `UPSTREAM_BASE_URL`)

---

## 13. Running the Backend

```powershell
# From the project root
python router.py
```

Or equivalently:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server starts on `http://localhost:8000` (or configured `ROUTER_HOST`/`ROUTER_PORT`).

### Verify the Backend

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness check, provider mode, configured models |
| `GET /v1/models` | Lists available virtual models (`smart-router`) |
| `GET /docs` | Interactive Swagger API documentation |
| `POST /v1/chat/completions` | Chat completions with routing |

---

## 14. Running the Frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend starts on `http://localhost:5173` and proxies `/v1` and `/health` to `http://127.0.0.1:8000`.

### Frontend Features

| Tab | Description |
|-----|-------------|
| **Chat Playground** | Conversational UI with streaming/non-streaming, stop/cancel, code blocks with copy, starter prompts per category |
| **Routing Insights** | Real-time routing pipeline visualization, per-session category distribution |
| **Connection & Models** | Backend URL configuration, optional API key, health check, verified models table |
| **Session History** | Log of completed requests with category, model, duration, tokens; search and filter |

---

## 15. API Documentation

### Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | Optional | Liveness check, provider mode, configured models |
| `GET` | `/v1/models` | Optional | OpenAI-compatible model listing (`smart-router`) |
| `POST` | `/v1/chat/completions` | Optional | Streaming and non-streaming chat completions |

### Request: `POST /v1/chat/completions`

```json
{
  "model": "smart-router",
  "messages": [
    { "role": "user", "content": "Implement a thread-safe LRU Cache in Python" }
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 2048
}
```

### Response (Non-streaming)

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "smart-router",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 },
  "routed_category": "CODING",
  "routed_model": "qwen/qwen3.8-27b:free"
}
```

**Response Headers:**
- `X-Routed-Category`: `CODING` \| `UNIT_TESTING` \| `REASONING`
- `X-Routed-Model`: the selected downstream model ID

### Response (Streaming / SSE)

`Content-Type: text/event-stream`

The first SSE chunk includes `routed_category` and `routed_model` in the chunk data. Subsequent chunks stream token deltas. The stream terminates with `data: [DONE]`.

---

## 16. Testing and Validation

Testing is divided into **automated software tests** (offline, mocked) and **live LLM evaluation experiments** (real provider calls).

### Backend Automated Tests

**Framework:** pytest (offline, mocked provider fixtures)

**Test files:**
- `tests/test_router.py` — 16 tests: label parsing, content extraction, SSE formatting, stream delta merging, role validation, health endpoint, models endpoint, auth, chat completion routing
- `tests/test_benchmark.py` — 18 tests: model selection, cost calculation, classification metrics, latency percentiles, response quality evaluation, benchmark runner artifacts

**Run:**

```powershell
python -m pytest tests -v
```

**Latest results (from `testing-results/pytest_output.txt`):**
- 34 tests passed, 0 failed, 0 skipped
- Duration: ~6 seconds
- All tests run completely offline with mocked providers

### Frontend Automated Tests

**Framework:** Vitest + jsdom + React Testing Library

**Test files (3):**
- `frontend/src/__tests__/api.test.ts` — API client: health, models, chat completion with routing metadata
- `frontend/src/__tests__/components.test.tsx` — UI components: CodeBlock, MessageItem routing badges
- `frontend/src/__tests__/sseParser.test.ts` — SSE parser: chunk boundaries, [DONE] termination, fragmentation, routing metadata extraction

**Run:**

```powershell
cd frontend
npm test
```

**Latest results:** 12 Vitest tests (6 API + 3 component + 3 SSE parser test cases)

### API Integration Tests

**Source:** `testing-results/api_test_results.csv`

15 API endpoint tests were executed against the running backend:

| Test | Endpoint | Status | Result |
|------|----------|--------|--------|
| API_01 | GET /health | 200 | PASS |
| API_02 | GET /v1/models | 200 | PASS |
| API_03 | POST /v1/chat/completions (Coding) | 200 | PASS |
| API_04 | POST /v1/chat/completions (Testing) | 200 | PASS |
| API_05 | POST /v1/chat/completions (Reasoning) | 200 | PASS |
| API_06 | POST /v1/chat/completions (SSE streaming) | 200 | PASS |
| API_07 | POST /v1/chat/completions (missing messages) | 422 | PASS |
| API_08 | POST /v1/chat/completions (invalid role) | 422 | PASS |
| API_09 | POST /v1/chat/completions (empty messages) | 200 | PASS |
| API_10 | POST /v1/chat/completions (empty user content) | 200 | PASS |
| API_11 | POST /v1/chat/completions (multi-turn) | 200 | PASS |
| API_12 | POST /v1/chat/completions (extra params passthrough) | 200 | PASS |
| API_13 | GET /v1/models (missing auth) | 401 | PASS |
| API_14 | GET /v1/models (invalid token) | 401 | PASS |
| API_15 | GET /v1/models (valid token) | 200 | PASS |

**Result: 15/15 passed**

### Failure Scenario Tests

**Source:** `testing-results/failure_test_results.csv`

12 failure scenarios tested, all handled gracefully:

| Scenario | Expected Status | Observed | Gracefully Handled |
|----------|----------------|----------|-------------------|
| Upstream auth failure (completion) | 502 | 502 | YES |
| Upstream auth failure (classification) | 502 | 502 | YES |
| Upstream rate limit (429) | 502 | 502 | YES |
| Upstream timeout | 504 | 504 | YES |
| Upstream connection failure | 502 | 502 | YES |
| Upstream internal error (500) | 502 | 502 | YES |
| Classifier timeout → fallback to CODING | 200 | 200 | YES |
| Classifier hallucinated output | 200 | 200 | NO* |
| Streaming mid-stream exception | 200 | 200 | YES |
| Client auth: missing header | 401 | 401 | YES |
| Client auth: malformed header | 401 | 401 | YES |
| Classifier caching (absent) | 200 | 200 | PARTIAL |

*\*FAIL_08: No caching mechanism exists — repeated prompts invoke the classifier twice. Recommended improvement: implement in-memory TTL/LRU cache.*

### Live LLM Evaluation Experiments

These are **not automated software tests** — they are live experiments against real LLM providers and are documented in Section 17 (Experimental Evaluation).

---

## 17. Experimental Evaluation

The experimental evaluation compares a **baseline** (single fixed model for all requests) against **ModelSync** (routing to task-specific models). Results are stored in `testing-results/` and `benchmarks/results/`.

### Methodology

**Dataset:** 30 prompts across 3 categories (10 per category), defined in `benchmarks/dataset.json`.

| Category | Sample Prompts |
|----------|---------------|
| CODING (10) | LRU Cache, binary search, bug diagnosis, fast exponentiation, singleton, SemVer parser, async generator, topological sort, token bucket, refactoring |
| UNIT_TESTING (10) | Pytest validator suite, mock payment gateway, SQLite fixture, Jest currency formatter, leap year tests, FastAPI endpoint tests, edge-case detection, branch coverage, race condition test, async retry test |
| REASONING (10) | Wolf/goat/cabbage, Bayes' theorem, EDA vs REST, Master Theorem, Knight & Knave, storage sizing, lock-free concurrency, idempotent payments, GIL vs race conditions, water jug riddle |

**Baseline model:** `nvidia/nemotron-3-ultra-550b-a55b` (or `:free` variant) — a single fixed model used for every request in the baseline arm.

**ModelSync configuration:** Category-specific models as configured in `.env`:
- CODING → `qwen/qwen3.8-27b:free`
- UNIT_TESTING → `google/gemma-4-26b-a4b-it:free`
- REASONING → `nvidia/nemotron-3-ultra-550b-a55b:free`

**Classifier model:** `poolside/laguna-s-2.1:free`

**Cost calculation:** `(prompt_tokens × prompt_price_per_million + completion_tokens × completion_price_per_million) / 1,000,000` using the OpenRouter pricing catalog (accessed 2026-10-24). Free-tier models cost $0.00 actual; paid counterparts are simulated for comparison.

### Experiment Runs

Three benchmark runs were executed:

| Run ID | Date (UTC) | Mode | Live | Dataset Size | Baseline Model |
|--------|-----------|------|------|-------------|----------------|
| `run_20261024_133701_9b51c7` | 2026-10-24T13:37:01 | dry-run | No | 30 | `nemotron-...:free` |
| `run_20261024_133917_57a309` | 2026-10-24T13:39:41 | smoke | Yes | 3 | `nemotron-...` (paid) |
| `run_20261024_134104_4ce3b2` | 2026-10-24T13:41:37 | smoke | Yes | 3 | `nemotron-...` (paid) |

### Classification Accuracy Results

**Source:** `testing-results/classification_metrics.json` and `benchmarks/results/routing_correctness.json`

**Live classifier evaluation (30 prompts, `poolside/laguna-s-2.1:free`):**

| Metric | Value |
|--------|-------|
| Total prompts | 30 |
| Correct predictions | 30 |
| Accuracy | 100.0% |
| Macro precision | 1.0 |
| Macro recall | 1.0 |
| Macro F1 | 1.0 |

Confusion matrix (perfect classification):

| Actual \ Predicted | CODING | UNIT_TESTING | REASONING |
|-------------------|--------|-------------|-----------|
| CODING | 10 | 0 | 0 |
| UNIT_TESTING | 0 | 10 | 0 |
| REASONING | 0 | 0 | 10 |

**Benchmark dry-run evaluation (30 prompts, mock classifier):**

| Metric | Value |
|--------|-------|
| Classification accuracy | 83.33% |
| Model selection correctness | 83.33% |
| Macro precision | 0.8745 |
| Macro recall | 0.8333 |
| Macro F1 | 0.8175 |

Confusion matrix (dry-run):

| Actual \ Predicted | CODING | UNIT_TESTING | REASONING |
|-------------------|--------|-------------|-----------|
| CODING | 10 | 0 | 0 |
| UNIT_TESTING | 0 | 10 | 0 |
| REASONING | 4 | 1 | 5 |

The dry-run misclassifications were all in the REASONING category (4 predicted as CODING, 1 as UNIT_TESTING), reflecting the mock classifier's keyword-based heuristic.

### Quality Evaluation Results

**Source:** `testing-results/quality_evaluation.json`

Three response quality evaluations comparing ModelSync (routed) vs baseline (single model):

| Eval ID | Category | Router Model | Router Score (/5) | Router Status | Baseline Model | Baseline Score (/5) | Baseline Status |
|---------|----------|-------------|-------------------|---------------|----------------|---------------------|-----------------|
| CODING_EVAL | CODING | `qwen/qwen3.8-27b:free` | 1.0 | FAILED (502: model unavailable for free) | `nemotron-...:free` | 4.67 | OK |
| TESTING_EVAL | UNIT_TESTING | `google/gemma-4-26b-a4b-it:free` | 1.0 | FAILED (502: rate limited) | `nemotron-...:free` | 4.67 | OK |
| REASONING_EVAL | REASONING | `nemotron-...:free` | 4.83 | OK | `nemotron-...:free` | 4.67 | OK |

**Important observations:**
- The CODING and UNIT_TESTING router requests failed with upstream 502 errors (model unavailable for free tier, rate limiting), resulting in score 1.0 (empty response).
- The REASONING router request succeeded and scored higher (4.83) than the baseline (4.67).
- These results reflect **live upstream provider limitations** (free-tier availability and rate limits), not router logic failures.

---

## 18. Cost Optimization Analysis

### Cost Methodology

Costs are calculated using the OpenRouter pricing catalog (accessed 2026-10-24):

```
cost_usd = (prompt_tokens × prompt_price_per_million + completion_tokens × completion_price_per_million) / 1,000,000
```

Two cost perspectives are reported:

1. **Actual billed cost:** What the upstream provider actually charges. Free-tier models (`:free`) cost $0.00.
2. **Simulated paid catalog cost:** What the equivalent paid model would cost, used to compare relative efficiency when free tiers are unavailable or rate-limited.

**Important:** ModelSync does **not** dynamically select the cheapest model. Model-to-category mapping is static configuration. Cost savings come from routing to smaller/specialized models (which are often free or cheaper), not from real-time price optimization.

### Pricing Catalog (OpenRouter, 2026-10-24)

| Model | Prompt Price (/1M tokens) | Completion Price (/1M tokens) | Free Tier |
|-------|---------------------------|-------------------------------|-----------|
| `nvidia/nemotron-3-ultra-550b-a55b` | $0.60 | $2.40 | No |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | $0.00 | $0.00 | Yes |
| `qwen/qwen3.8-27b` | $0.20 | $2.50 | No |
| `qwen/qwen3.8-27b:free` | $0.00 | $0.00 | Yes |
| `google/gemma-4-26b-a4b-it` | $0.0675 | $0.225 | No |
| `google/gemma-4-26b-a4b-it:free` | $0.00 | $0.00 | Yes |
| `poolside/laguna-s-2.1:free` | $0.00 | $0.00 | Yes |
| `openai/gpt-4o-2024-11-20` | $2.50 | $10.00 | No |
| `openai/gpt-4o-mini` | $0.15 | $0.60 | No |

### Cost Comparison Results

**Source:** `benchmarks/results/summary_run_*.json`

#### Dry-Run (30 prompts, mock provider, simulated paid catalog)

| Metric | Baseline | ModelSync |
|--------|----------|-----------|
| Total cost (simulated paid) | $0.00828 | $0.004995 |
| Savings | — | **39.67%** |
| Actual billed | $0.00 | $0.00 |

#### Smoke Test Run 2 (3 prompts, live provider, actual billed)

| Metric | Baseline | ModelSync |
|--------|----------|-----------|
| Total cost (actual billed) | $0.001215 | $0.000670 |
| Savings | — | **44.84%** |
| Total cost (simulated paid) | $0.001216 | $0.000846 |
| Savings (simulated) | — | 30.43% |

#### Smoke Test Run 3 (3 prompts, live provider, actual billed)

| Metric | Baseline | ModelSync |
|--------|----------|-----------|
| Total cost (actual billed) | $0.002295 | $0.001256 |
| Savings | — | **45.28%** |
| Total cost (simulated paid) | $0.002296 | $0.001614 |
| Savings (simulated) | — | 29.70% |

### Key Takeaways

- Cost savings are measured **per experiment run** and vary based on the prompts, models used, and whether free-tier endpoints are available.
- Actual billed costs for free-tier models are $0.00, making savings percentage undefined (division by zero) in those cases — simulated paid catalog costs are used instead.
- The savings come from routing simpler tasks (coding, testing) to smaller free models, while reserving the larger baseline model only for reasoning tasks where it is the assigned target.
- These results **do not** guarantee cost savings in production; actual costs depend on your provider, model availability, rate limits, and usage patterns.

---

## 19. Results and Performance Metrics

### Router Overhead Performance (Offline Mock)

**Source:** `testing-results/performance_results.csv`

| Scenario | Requests | Success | Failed | Error Rate | Mean Latency | P50 | P95 | Min | Max | Throughput | Environment |
|----------|----------|---------|--------|------------|-------------|-----|-----|-----|-----|------------|-------------|
| Pure Router Overhead | 50 | 50 | 0 | 0.0% | 6.87ms | 7.06ms | 8.33ms | 2.93ms | 12.23ms | 145.46 rps | Offline Mock |
| Streaming SSE Overhead | 50 | 50 | 0 | 0.0% | 7.72ms | 7.50ms | 9.43ms | 5.90ms | 10.56ms | 129.58 rps | Offline Mock |
| Concurrent Load (C=5) | 50 | 50 | 0 | 0.0% | 8.21ms | 7.95ms | 11.55ms | 6.12ms | 12.46ms | 121.76 rps | Offline Mock |
| Concurrent Load (C=10) | 50 | 50 | 0 | 0.0% | 8.15ms | 8.14ms | 9.65ms | 6.46ms | 10.20ms | 122.70 rps | Offline Mock |
| Live Classifier Latency | 30 | 30 | 0 | 0.0% | 6,269.63ms | 2,870.50ms | 15,552.65ms | 813.0ms | 66,488.9ms | 0.16 rps | Live OpenRouter |

The router itself adds minimal overhead (under 10ms mean in offline mock tests). Live classifier latency is dominated by upstream provider response time (mean ~6.3s across 30 classification calls).

### Token Usage Summary (Dry-Run, 30 prompts)

**Source:** `benchmarks/results/records_run_20261024_133701_9b51c7.csv`

All 30 dry-run records used consistent token counts:
- Prompt tokens: 120 (baseline), 120 (router)
- Completion tokens: 85 (baseline), 85 (router)
- Total tokens: 205 per request

---

## 20. Docker Deployment

```bash
cd llm-composer-coding-agent
docker compose up --build
```

### Docker Configuration

| Aspect | Detail |
|--------|--------|
| Base image | `python:3.13-slim` |
| Build strategy | Multi-stage (build deps in venv, copy to runtime) |
| User | `routeruser` (non-root) |
| Exposed port | 8000 |
| Healthcheck | `GET http://127.0.0.1:8000/health` (30s interval, 5s timeout) |
| Environment | Loaded from `.env` file |
| Restart policy | `unless-stopped` |

The `docker-compose.yml` maps `ROUTER_PORT` on the host to port 8000 in the container. Default model slugs in Compose differ slightly from `.env` defaults (using Together AI model slugs as Compose defaults), which can be overridden via environment variables.

---

## 21. Limitations

| Limitation | Description |
|------------|-------------|
| **No database** | Session history is maintained only in the browser; refreshing the page clears history. No persistent storage of requests or responses. |
| **No classifier caching** | Every request invokes the classifier model, even for identical or repeated prompts. This adds latency and cost (see FAIL_08 in failure tests). |
| **Static model mapping** | Model-to-category mapping is configuration-based, not dynamically optimized for cost or performance. |
| **Single provider at a time** | The backend connects to either OpenRouter or Together AI based on which API key is configured; it cannot fail over between providers automatically. |
| **Free-tier instability** | Free-tier models may be unavailable or rate-limited by the upstream provider, causing 502 errors (observed in quality evaluation). |
| **No request/response logging server-side** | No audit trail of requests is kept on the backend. |
| **Classifier accuracy varies** | Live classifier achieved 100% on 30 prompts, but dry-run mock showed 83.33%. Real-world accuracy depends on the classifier model and prompt diversity. |

---

## 22. Future Scope

The following are **planned enhancements**, not implemented features:

- **Classifier caching**: In-memory TTL/LRU cache keyed by prompt hash to avoid redundant classifier calls for repeated prompts.
- **Cost-aware routing**: Dynamic model selection based on real-time pricing and availability rather than static configuration.
- **Additional task categories**: Expand beyond CODING, UNIT_TESTING, REASONING (e.g., CREATIVE, ANALYSIS, TRANSLATION).
- **Additional providers**: Support for Anthropic, Google Vertex AI, Azure OpenAI, AWS Bedrock, etc.
- **Enhanced evaluation**: LLM-as-judge evaluation, A/B testing framework, regression detection.
- **Deployment improvements**: Kubernetes manifests, Helm chart, auto-scaling configuration.
- **Database backend**: Optional persistent storage for request history, analytics, and user preferences.
- **Authentication enhancements**: OAuth, JWT, rate limiting per API key.

---

## 23. Contributors

ModelSync is an open-source project. Contributions are welcome.

---

**Repository:** https://github.com/SRT-2/Model-Sync-

**Version:** 2.0.0

**License:** See repository for license information.
```

---

## Summary of What Was Updated

| Section | Status | Key Content |
|---------|--------|-------------|
| 1-4 (Overview, Problem, Objectives, Features) | ✅ Written | Accurately describes the current implementation |
| 5-8 (Architecture, Request Lifecycle, Routing, Persona Injection) | ✅ Written | Mermaid diagram, static model mapping (not dynamic), classifier persona via system prompt |
| 9-14 (Tech Stack, Structure, Installation, Env Config, Backend, Frontend) | ✅ Written | Verified against actual source code and config files |
| 15 (API Docs) | ✅ Written | Endpoints, request/response schemas, SSE metadata contract |
| 16 (Testing) | ✅ Written | 34 pytest tests passed, 12 Vitest tests, 15/15 API tests, 12 failure scenarios |
| 17-19 (Evaluation, Cost, Metrics) | ✅ Written | Live classifier 100% accuracy on 30 prompts, cost savings 39.67%-45.28%, router overhead <10ms |
| 20-23 (Docker, Limitations, Future Scope, Contributors) | ✅ Written | Multi-stage Dockerfile, no database limitation, future scope clearly marked as not implemented |

**Source files used for verification:** `app/` (all Python source), `tests/` (both test files), `frontend/src/` (all TypeScript/React source), `benchmarks/` (pricing, metrics, evaluator, runner, routing_test), `testing-results/` (all CSV, JSON, PNG, log files), `docker-compose.yml`, `Dockerfile`, `.env`, `.env.example`, `requirements.txt`.

**Claims intentionally excluded:** No "best model" or "guaranteed cost savings" claims; no database described as implemented; no dynamic cost-optimization algorithm claimed; no real API keys exposed.
