# ModelSync — LLM Smart Router v2

An OpenAI-compatible FastAPI routing engine and modern React + TypeScript dashboard.
It analyzes each incoming prompt with a lightweight classifier, identifies the task category (**`CODING`**, **`UNIT_TESTING`**, or **`REASONING`**), and dynamically forwards the request to the optimal downstream model — while clients and user interfaces connect seamlessly to a single virtual model: `smart-router`.

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │    Client / ModelSync Web Dashboard    │
                      └───────────────────┬────────────────────┘
                                          │ POST /v1/chat/completions (model: smart-router)
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    FastAPI Router Server (:8000)       │
                      └───────────────────┬────────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        │ Step 1: Lightweight Classifier    │
                        │ (Prompt analysis & intent match)  │
                        └─────────────────┬─────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼                             ▼                             ▼
   [ Category: CODING ]         [ Category: TESTING ]        [ Category: REASONING ]
   zai-org / Qwen / Cohere      moonshotai / Gemma           Qwen-Max / Nemotron
            │                             │                             │
            └─────────────────────────────┼─────────────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ Upstream Provider (OpenRouter/Together)│
                      └────────────────────────────────────────┘
```

---

## Quickstart (Windows PowerShell)

### 1. Start Backend (FastAPI)

```powershell
# Navigate to the project root
cd llm-composer-coding-agent

# Configure environment variables
Copy-Item .env.example .env     # Edit .env and enter your OPENROUTER_API_KEY or TOGETHER_API_KEY

# Run server
python router.py
```

- Backend API: `http://localhost:8000`
- Interactive API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Virtual Models: `http://localhost:8000/v1/models`

---

### 2. Start Frontend (React + TypeScript + Vite)

In a second PowerShell terminal:

```powershell
cd llm-composer-coding-agent\frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

- Frontend UI: `http://localhost:5173`

---

## Web Dashboard Features

1. **Chat Playground**:
   - OpenAI-compatible conversational UI with streaming and non-streaming support.
   - Enter-to-send and Shift+Enter for newlines.
   - Stop/cancel streaming with `AbortController` signal cleanup.
   - Safe Markdown rendering and syntax-highlighted code blocks with one-click copy.
   - Dynamic routing badge indicating whether the prompt was classified as `CODING`, `UNIT_TESTING`, or `REASONING`, along with the selected downstream model and round-trip duration.
   - One-click starter prompts for testing each routing category.
   - Configurable inference parameters (Streaming toggle, Temperature, Max Tokens).

2. **Routing Insights**:
   - Real-time pipeline architecture diagram.
   - Live backend routing table fetched directly from `/health`.
   - Current session distribution metrics (percentage of coding vs testing vs reasoning tasks).

3. **Connection & Models**:
   - Configurable backend base URL (default `http://localhost:8000`).
   - One-click `/health` connectivity test and detailed runtime status.
   - Inbound Bearer token authentication input (kept safely in memory only; clearly separated from upstream provider keys).
   - Live table of verified active models from `/v1/models`.

4. **Session Request History**:
   - Detailed log of completed requests with category tags, downstream model, response duration, and token usage.
   - Search by keyword and filter by routing category (`CODING`, `UNIT_TESTING`, `REASONING`).
   - Copy prompt / copy response buttons.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check, provider mode, and configured downstream models |
| `GET` | `/v1/models` | OpenAI-compatible model listing exposing `smart-router` |
| `POST` | `/v1/chat/completions` | Streaming (SSE) and non-streaming chat completions with routing metadata |

### Routing Metadata Contract
Responses from `/v1/chat/completions` include backward-compatible routing metadata:
- Response Headers: `X-Routed-Category` (`CODING` \| `UNIT_TESTING` \| `REASONING`) and `X-Routed-Model` (the selected target model ID).
- Response JSON: fields `routed_category` and `routed_model`.
- Streaming SSE: Initial metadata chunk emits `routed_category` and `routed_model` before token stream begins.

---

## Testing

### Run Backend Unit Tests (Offline)
```powershell
cd llm-composer-coding-agent
python -m pytest tests -v
```
*(All 16 unit tests run completely offline with mocked provider fixtures).*

### Run Frontend Unit Tests
```powershell
cd llm-composer-coding-agent\frontend
npm test
```
*(12 Vitest tests covering SSE parser chunk boundaries, [DONE] termination, API client calls, and UI components).*

### Production Build
```powershell
cd llm-composer-coding-agent\frontend
npm run build
```

---

## Docker Quickstart

```bash
cd llm-composer-coding-agent
docker compose up --build
```
The Docker container builds a multi-stage Python 3.13-slim image running as a non-root user (`routeruser`) with an automated healthcheck.

---

## Project Layout

```
llm-composer-coding-agent/
├── app/
│   ├── main.py            # FastAPI app factory, CORS, and lifespan
│   ├── config.py          # Pydantic Settings (.env loader)
│   ├── state.py           # Singleton state management
│   ├── schemas.py         # Request/response models & role validation
│   ├── routing.py         # Classifier prompts, label parser & SSE formatter
│   ├── providers/
│   │   ├── base.py        # LLMProvider abstract base class
│   │   └── together.py    # OpenAI-compatible provider (OpenRouter & Together)
│   └── api/
│       ├── deps.py        # Constant-time Bearer token authentication
│       └── routes.py      # /health, /v1/models, /v1/chat/completions
├── frontend/
│   ├── src/
│   │   ├── components/    # Playground, Insights, Settings, History, CodeBlock
│   │   ├── context/       # AppContext state & API orchestration
│   │   ├── services/      # API client & robust SSEStreamParser
│   │   ├── styles/        # Professional developer design system (dark/light)
│   │   └── __tests__/     # Vitest unit tests for SSE, API, and UI
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── tests/
│   └── test_router.py     # 16 offline pytest test cases
├── router.py              # CLI entry point
├── Dockerfile             # Multi-stage production container
├── docker-compose.yml     # Compose service specification
└── README.md              # Project documentation
```
