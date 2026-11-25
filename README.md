# FilingLens: A RAG and Tool-calling Agent for Stock Investment Research

## Project structure

```
FilingLens/
├── backend/
│   ├── app                            
│   │   ├── main.py                    # FastAPI server: /health, /health/db, /chat
│   │   ├── config.py                  # Loads settings from .env
│   │   ├── db.py                      # Opens connection to Supabase
│   │   ├── schemas.py                 # Data models: companies, sections, chunks, metrics
│   │   ├── fiscal.py                  # Computes fiscal years and quarters
│   │   ├── logger.py                  # Logs each pipeline step
│   │   ├── pipeline.py                # Answers a question by running each step in order
│   │   ├── steps/                     # Pipeline steps: rewrite, route, pick tool, run tool, answer, check citations
│   │   ├── ingest/                    # Builds data: filings -> sections -> chunks, metrics, Supabase
│   │   ├── retrieval/                 # Embeds chunks and searches them
│   │   └── eval/                      # Scores the full pipeline and each part
│   └── tests                          # Pytest tests
├── data/
│   ├── processed/                     # Downloaded filings (gitignored)
│   ├── raw/                           # Built data (gitignored)
│   └── companies.json                 # Companies covered (ticker, CIK, fiscal year end)
├── eval/
│   ├── sets                           # Evaluation questions with answers 
│   └── results
│       ├── <set>-<run>.json           # Scores for each test run
│       ├── end-to-end-stats.json      # Speed and cost of full pipeline
│       ├── contexts                   # Sources each answer was written from
│       ├── judgements                 # LLM judge scores
│       ├── predictions                # Raw outputs for each test run
│       └── spotcheck                  # Human grading sheet, grades, and answer key
└── frontend
```

## What the agent does
FilingLens answers questions about 10 large tech companies (AAPL, MSFT, GOOGL, AMZN, NFLX, NVDA, TSLA, CSCO, ORCL, and ADBE) using their SEC filings between fiscal years 2016 and 2026. Each question goes through a pipeline:
1. Follow-up questions are rewritten into full questions based the chat history. Fpr example, "what was their net revenue that year?" is rewritten to "what was Amazon's net revenue in 2020?"
2. An LLM router decides the question type:
   - Narrative: asks about filing text, such as risks factors or MD&A
   - Metric: asks about reported numbers, such as revenue or EPS
   - Hybrid: asks about filing text and reported numbers
   - Ambiguous: too vague to answer
   - Out-of-scope: asks for investment advice or price predictions, or asks about companies, years, or metrics not covered

   In addition, the LLM router decides the company, fiscal year, and period the question is about, which is used to filter search. 
3. If the question is narrative, metric, or hybrid, the system searches only the filing chunks that match the router's filters and returns the top-20 using consine similarity.
4. If the router decides the question type is metric or hybrid, an LLM tool selector decides one of three metric tools and fills in its arguments. The three tools are:
   - `get_metric`: one company, one period
   - `compare_companies`: two or more companies, one period
   - `get_metric_series`: one company, a range of years

   Tools cover 7 metrics: revenue, net income, operating income, operating cash flow, R&D expense, total assets, diluted EPS. The filing search and metric tools run in parallel.
5. The generator receives the original question and retrieved evidence from the filing search and/or metric tools, and writes the answer with numbered citations linked to real SEC filings. If the question was marked ambiguous, then the system writes a clarifying question. If the question was marked out-of-scope, then the system refuses to answer.
6. The citation checker flags any numbers in the answer that do not appear in the cited sources. Flagged answers are still sent, but marked.

## Target users
Long-term stock investment plays a critical role in accomplishing future financial goals such as educational funding, retirement, or placing a downpayment on a house. Before individuals risk their own money into a potential stock, it is important for them to learn about the company's performance. There are two methods to do so: technical analysis and fundamental analysis. Technical analysis examines the change in stock prices over time, while fundamental analysis investigates the history, financial health, and management of companies. Many investors rely on technical analysis because price charts and market trends are relatively easy to interpret, whereas fundamental analysis means reviewing long SEC filings.

FilingLens targets long-term investors who wish to perform fundamental analysis without reading lengthy SEC filings and risking hallucinated answers from general chatbots. Users simply send a question regarding a company's 10-Q or 10-K reports and receive cited answers that help them make informed decisions. 

The scope of this project includes 10 large tech companies (AAPL, MSFT, GOOGL, AMZN, NFLX, NVDA, TSLA, CSCO, ORCL, and ADBE) between fiscal years 2016 and 2026. Additionally, questions that ask for investment advice or price predictions are considered out of scope and get refused.

## Setup
**Requirements:** uv (``brew install uv``) and Node 18+
1. Clone the repo:
   ```bash
   git clone https://github.com/jeffreyhoang/A-Hybrid-Machine-Learning-and-Classical-Planning-System-for-Multi-Robot-Warehouse-Coordination.git
   ```
2. Create your env file:
   ```bash
   cp .env.example .env
   ```
   `DATABASE_URL` is already filled in with a read-only key.
   Add your own keys:
   ```
   OPENAI_API_KEY=...
   ANTHROPIC_API_KEY=... (optional)
   SEC_USER_AGENT=... (optional)
   ```
3. Install the backend (uv install Python 3.12 automatically):
   ```bash
   cd RxCite-Cited-Answers-to-Consumer-Medication-Questions/backend
   uv sync
   ```
4. Install the frontend:
   ```bash
   cd ../frontend
   npm install
   ```
5. Run the app:

   Use two terminals, both starting from the repo root.

   **Terminal 1 (backend):**
   ```bash
   cd backend 
   uv run uvicorn app.main:app --reload --port 8000
   ```
   **Terminal 2 (frontend)**
   ```bash
   cd frontend
   npm run dev
   ```

   Open http://localhost:5173 and ask a question.





## Data sources and tools

## Evaluation
| col1 | col2 |
|--------------|-----------|
| val | val |


## Safety and robustness

## Limitations

## Future improvements