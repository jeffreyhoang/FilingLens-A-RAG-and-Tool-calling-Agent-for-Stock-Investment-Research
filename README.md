# FilingLens: A Hybrid RAG and Tool-calling Agent for Stock Investment Research

Ask questions about a company's 10-K and 10-Q filings in natural language and get cited text and chart answers.

## Project Structure

```
.
├── backend/                 # FastAPI app (Python 3.12, managed with uv)
│   ├── app/
│   │   ├── main.py          # FastAPI app and routes (/health, /health/db)
│   │   └── config.py        # Settings loaded from .env
│   ├── pyproject.toml       # Dependencies
│   └── uv.lock              # Locked dependency versions
├── frontend/                # Next.js chat UI
├── data/
│   ├── raw/                 # Downloaded SEC filings (not committed)
│   └── processed/           # Cleaned sections, chunks, and metrics
├── eval/                    # Test sets, evaluation harness, and results
├── notebooks/               # Data exploration and experiments
├── plans/                   # Project plan and per-ticket step files
└── .env.example             # Required environment variables (copy to .env)
```

The ingestion, retrieval, router, tools, and generator code will be added under `backend/app/` in later tickets.

## Setup

**Prerequisites:** [uv](https://docs.astral.sh/uv/) (`brew install uv`) and a [Supabase](https://supabase.com) project. uv installs Python 3.12 for you.

1. **Install dependencies**
   ```
   cd backend
   uv sync
   ```
2. **Create `.env`** at the repo root and fill in the values (never commit it):
   ```
   cp .env.example .env
   ```

   | Variable | Where to find it |
   |---|---|
   | `SUPABASE_URL` | Supabase dashboard > Project Settings > API |
   | `SUPABASE_KEY` | Same page, the **secret** key (server only) |
   | `DATABASE_URL` | Connect > URI > **Session pooler** connection string |
   | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Provider dashboards (needed from T7) |

3. **Enable pgvector** once per Supabase project, in the SQL editor:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;
   ```
4. **Run the backend** from `backend/`:
   ```
   uv run uvicorn app.main:app --reload
   ```
   Check `http://localhost:8000/health` (`{"status": "ok"}`) and `http://localhost:8000/health/db` (returns the pgvector version). Interactive docs are at `/docs`.

**Lint and format** with [Ruff](https://docs.astral.sh/ruff/) (from `backend/`):
- `uv run ruff check .` lints: flags likely bugs and style issues, such as unused imports or undefined names. Add `--fix` to auto-fix what it can.
- `uv run ruff format .` formats: rewrites code to a consistent style without changing what it does.
