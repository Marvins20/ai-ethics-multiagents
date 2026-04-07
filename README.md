# AI Ethics Multi-Agent Auditor

A multi-agent AI ethics auditing system that analyzes AI projects for ethical risks, cross-references historical incidents, and evaluates compliance with the Brazilian AI Ethics Framework (PL 2338/2023).

Built with **LangGraph**, **FastAPI**, **PostgreSQL**, and **Redis**. Exposes a REST API for a React frontend, with real-time agent progress streaming via Server-Sent Events.

---

## Architecture Overview

```
React Frontend
      │
      │ REST + SSE
      ▼
FastAPI (src/app.py)
      │
      ├── POST /api/v1/projects/{id}/audits ──► ARQ Job Queue (Redis)
      │                                               │
      │                                               ▼
      │                                        Worker (src/worker/)
      │                                        LangGraph Pipeline
      │                                        ├── Project Analyst
      │                                        ├── Risk Agent       ─► Chroma (risk DB)
      │                                        ├── Incident Agent   ─► Chroma (incidents) + DuckDB
      │                                        ├── Framework Agent  ─► Chroma (PL 2338/2023)
      │                                        └── Final Classifier
      │                                               │
      │                                        Checkpoints saved to PostgreSQL (LangGraph)
      │
      └── GET /api/v1/audits/{id}/stream ──► Redis Pub/Sub ──► SSE to client
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Google Gemini 2.5 Pro |
| Embeddings | Google `models/gemini-embedding-001` |
| Vector store | Chroma (local, persisted to disk) |
| Relational DB | PostgreSQL (Supabase free tier) |
| Job queue | ARQ + Redis (Upstash free tier) |
| Web framework | FastAPI + uvicorn |
| Migrations | Alembic |

---

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (package manager)
- A **Supabase** project — [supabase.com](https://supabase.com) (free tier)
- A **Redis** instance — [Upstash](https://upstash.com) free tier, or run locally with Docker
- A **Google AI API key** — [aistudio.google.com](https://aistudio.google.com)
- *(Optional)* A **LangSmith** API key for tracing — [smith.langchain.com](https://smith.langchain.com)

---

## Local Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/Marvins20/ai-ethics-multiagents.git
cd ai-ethics-multiagents
uv sync
```

### 2. Configure environment variables

Copy the example and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# ── Google Gemini ─────────────────────────────────────────────
GOOGLE_API_KEY=your_google_api_key_here
EMBEDDING_MODEL_NAME=models/gemini-embedding-001

# ── PostgreSQL (Supabase) ─────────────────────────────────────
# Async URL used by FastAPI / SQLAlchemy at runtime
DATABASE_URL=postgresql+asyncpg://postgres:<password>@<host>:5432/postgres
# Sync URL used by Alembic for migrations (same credentials, different driver)
DATABASE_URL_SYNC=postgresql+psycopg://postgres:<password>@<host>:5432/postgres

# ── Redis (Upstash or local) ──────────────────────────────────
REDIS_URL=redis://localhost:6379
# For Upstash use: rediss://default:<password>@<host>:6380

# ── JWT ───────────────────────────────────────────────────────
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY=your_random_secret_key_here

# ── LangSmith (optional tracing) ─────────────────────────────
LANGCHAIN_TRACING_V2=false
LANGSMITH_API_KEY=
LANGCHAIN_PROJECT=ai-ethics-multiagents

# ── Data paths ────────────────────────────────────────────────
AI_RISK_DATA_DIR=data/raw/ai_risk_database_v3.csv
```

> **Getting your Supabase URLs:** go to your project → Settings → Database → Connection string. Use the **URI** format. Replace the `postgresql://` prefix with `postgresql+asyncpg://` for `DATABASE_URL` and `postgresql+psycopg://` for `DATABASE_URL_SYNC`.

### 3. Run Redis locally (if not using Upstash)

```bash
docker run -d -p 6379:6379 redis:alpine
```

### 4. Apply database migrations

```bash
uv run alembic upgrade head
```

This creates the `users`, `projects`, and `audit_jobs` tables. LangGraph checkpoint tables are created automatically on the first worker run.

### 5. Start the API server

```bash
uv run uvicorn src.app:app --reload
```

The API will be at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 6. Start the background worker

In a separate terminal:

```bash
uv run arq src.worker.settings.WorkerSettings
```

The worker processes audit jobs from the queue and runs the LangGraph pipeline.

> Both the API server and the worker must be running to submit and process audits.

---

## Data Ingestion

The vector databases (Chroma) are populated automatically on first startup from the CSV and PDF files in `data/raw/`. Subsequent startups skip re-ingestion if data is already present.

Required files in `data/raw/`:
```
data/raw/
  ai_risk_database_v3.csv    ← AI risk taxonomy
  incidents.csv              ← AI incident database
  reports.csv                ← Incident reports
  PL_2338-2023.pdf           ← Brazilian AI framework document
```

---

## Usage

### As an API

Once running, use the Swagger UI at `/docs` or any HTTP client.

**1. Register a user**
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "yourpassword"}'
```

**2. Log in and get a token**
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -F "username=user@example.com" \
  -F "password=yourpassword"
```

**3. Create a project**
```bash
curl -X POST http://localhost:8000/api/v1/projects \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "Hiring AI System", "description": "Resume screening tool"}'
```

**4. Submit an audit**
```bash
curl -X POST http://localhost:8000/api/v1/projects/<project_id>/audits \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"project_description": "An AI system that screens resumes and ranks candidates automatically."}'
```

**5. Stream agent progress (SSE)**
```bash
curl -N http://localhost:8000/api/v1/audits/<job_id>/stream \
  -H "Authorization: Bearer <token>"
```

**6. Get the completed result**
```bash
curl http://localhost:8000/api/v1/audits/<job_id> \
  -H "Authorization: Bearer <token>"
```

### As a CLI (original mode)

```bash
uv run ai-ethics-multiagents "Describe your AI project here"
# or interactively:
uv run ai-ethics-multiagents
```

Outputs `agent_output_<timestamp>.json` and `agent_output_<timestamp>.md`.

---

## Frontend Integration

FastAPI auto-generates an OpenAPI spec. Use it to generate a typed TypeScript client for your React frontend.

**Export the spec:**
```bash
curl http://localhost:8000/openapi.json -o openapi.json
```

**Generate TypeScript types** (in your frontend repo):
```bash
npm install openapi-fetch
npx openapi-typescript openapi.json -o src/api/schema.d.ts
```

**Use in React:**
```ts
import createClient from 'openapi-fetch';
import type { paths } from './api/schema';

const client = createClient<paths>({
  baseUrl: import.meta.env.VITE_API_URL
});

// Submit an audit
const { data } = await client.POST('/api/v1/projects/{project_id}/audits', {
  params: { path: { project_id } },
  body: { project_description: "..." }
});
```

**Stream agent progress:**
```ts
const es = new EventSource(`${import.meta.env.VITE_API_URL}/api/v1/audits/${jobId}/stream`);
es.onmessage = (e) => console.log(JSON.parse(e.data));
es.onerror = () => es.close();
```

Set the `VITE_API_URL` environment variable in your frontend to point to your deployed backend URL.

---

## Project Structure

```
src/
  app.py                    # FastAPI factory + lifespan (RAG warm-up)
  config.py                 # Settings via pydantic-settings (.env)
  graph.py                  # LangGraph workflow (build_graph())
  model.py                  # Gemini LLM singleton
  state.py                  # AgentState TypedDict + Pydantic output models
  main.py                   # CLI entry point

  agents/                   # LangGraph agent nodes
    project_analyst_agent.py
    supervisor_agent.py
    risk_agent.py
    incident_agent.py
    proprietary_framework_agent.py
    final_classifier_agent.py

  api/v1/                   # REST endpoints
    auth.py                 # POST /auth/register, POST /auth/login
    users.py                # GET/PUT /users/me
    projects.py             # CRUD /projects
    audits.py               # POST /projects/{id}/audits, GET /audits/{id}/stream

  db/                       # SQLAlchemy async ORM
    models/                 # User, Project, AuditJob

  core/
    security.py             # JWT + bcrypt

  schemas/                  # Pydantic request/response schemas

  worker/
    tasks.py                # ARQ task: runs LangGraph + publishes SSE events
    settings.py             # ARQ WorkerSettings

  services/                 # ETL services (Chroma ingestion, DuckDB)
  tools/rags/               # RAG tools (risk, incidents, framework)

alembic/                    # Database migrations
data/
  raw/                      # Source CSV and PDF files
  chroma/                   # Persisted vector store (auto-generated)
  duckdb/                   # Reports database (auto-generated)
```

---

## Deployment

Recommended: **Railway** (backend) + **Vercel** (frontend) + **Supabase** (PostgreSQL) + **Upstash** (Redis).

**Procfile** for Railway:
```
web: uvicorn src.app:app --host 0.0.0.0 --port $PORT --workers 1
worker: arq src.worker.settings.WorkerSettings
```

Set all `.env` variables as Railway environment variables. Mount a persistent volume at `/app/data/chroma` to preserve the Chroma vector store across deployments.

Add CORS to `src/app.py` before deploying:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
