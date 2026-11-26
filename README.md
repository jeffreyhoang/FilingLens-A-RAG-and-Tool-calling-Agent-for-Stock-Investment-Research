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
1. Follow-up questions are rewritten into full questions based the chat history. For example, "what was their net revenue that year?" is rewritten to "what was Amazon's net revenue in 2020?"
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
**Requirements:** uv (`brew install uv`) and Node 20.19+
1. Clone the repo:
   ```bash
   git clone https://github.com/jeffreyhoang/FilingLens-A-RAG-and-Tool-calling-Agent-for-Stock-Investment-Research.git
   cd FilingLens-A-RAG-and-Tool-calling-Agent-for-Stock-Investment-Research
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
3. Install the backend (uv installs Python 3.12 automatically):
   ```bash
   cd backend
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
   **Terminal 2 (frontend):**
   ```bash
   cd frontend
   npm run dev
   ```

   Open http://localhost:3000 and ask a question.

## Data sources and tools
**Data sources:** 
- **SEC EDGAR filings:** 10-K and 10-Q filings for the 10 companies, fiscal years 2016–2026. Filing text is split into sections and chunks for search.
- **SEC XBRL CompanyFacts API:** reported financial figures (`data.sec.gov/api/xbrl/companyfacts`) covering the 7 metrics.

**Tools:**
- **EdgarTools:** downloads filings from SEC Edgar and extracts its sections
- **OpenAI:** `gpt-5.4-mini` used as the router, tool selector, follow-up rewriter, and generator; `text-embedding-3-large` used as the embedder; `gpt-5.4` used as the LLM judge.
- **Supabase:** stores chunks, embeddings, and metrics, and runs dense search.
- **FastAPI:** the backend API.
- **React + Tailwind + Recharts:** the frontend and its visualizations.

## Models and configurations compared
| Step | Compared | Chosen |
|---|---|---|
| Embedding model | `text-embedding-3-large`, `text-embedding-3-small`, `bge-base`, `gte-modernbert` | `text-embedding-3-large` |
| Chunk size | 1,000, 2,000, 4,000 characters | 1,000 |
| Search filters | none, answer-key filters, router's filters | router's filters |
| Query rewrite before search | with and without | without |
| Reranker (`bge-reranker-v2-m3` on top 50) | with and without | without |
| Router | `gpt-5.4-mini`, `claude-haiku-4-5` | `gpt-5.4-mini` |
| Tool selector | `gpt-5.4-mini` only | — |
| Generator | `gpt-5.4-mini`, `claude-haiku-4-5` | `gpt-5.4-mini` |

## Evaluation
1. **Test sets**
   | Set | Questions | What it holds |
   |-----|-----------|------------------|
   | Retrieval | 60 | Evidence with ticker, year, and period |
   | Router | 50 | Correct route (10 per route) | 
   | Tool | 30 | Correct tool, arguments, and values |
   | Generation | 30 | Reference answer (22 answer, 4 clarify, 4 refuse) |
   | Safety | 30 | Expected behavior for tricky questions |

2. **Embedding models (2,000-char chunks)**
   | Model | Recall@10, no filters | Recall@10, filters |
   |-------|-----------------------|--------------------|
   | text-embedding-3-large | 41.7% | **73.3%** |
   | text-embedding-3-small | 40.0% | 66.7% |
   | bge-base | **45.0%** | 63.3% |
   | gte-modernbert | 41.7% | 55.0% |

   Models were compared on recall@10 with filters. The `text-embedding-3-large` model achieved the highest score and was used in the remaining experiments.

3. **Chunk size (`text-embedding-3-large`, same 20,000 characters sent to the generator)**
   | Chunk size | Overlap | Chunks sent | Recall, no filters | Recall, filters |
   |------------|---------|-------------|--------------------|-----------------|
   | 1,000 | 100 | 20 | **65.0%** | **83.3%** |
   | 2,000 | 200 | 10 | 41.7% | 73.3% |
   | 4,000 | 400 | 5 | 36.7% | 56.7% |

   Recall@k is measured at k = 20, 10, and 5 for 1,000, 2,000, and 4,000 chunks, respectively. This ensures the same ~20,000 characters are sent. 1,000-characters chunks with 100-character overlap achieved the highest score and was used in the remaining experiments.

4. **Retrieval setup (`text-embedding-3-large`, 1,000-char chunks)**
   | Setup | Recall@20 | MRR |
   |-------|-----------|-----|
   | No filters | 65.0% | 0.296 |
   | Answer-key filters | 83.3% | 0.473 |
   | Router's filters | 83.3% | 0.473 |
   | Router's filters + query rewrite* | 73.3% | 0.363 |
   | Answer-key filters + reranker | **85.0%** | **0.657** |

   *Run with an earlier router version, which scored 85.0% on recall@20 without the rewrite.

5. **Router, tool selector, and generator**
   | Step | Metric | `gpt-5.4-mini` | `claude-haiku-4-5` |
   |------|--------|--------------|-----------------|
   | Router | Route accuracy (50 questions) | 100.0% | 100.0% |
   | Router | Filters exactly right (60 questions) | 91.7% | 83.3% |
   | Tool selector | Right tool, arguments, values (30 questions) | 100.0% | Not tested |
   | Generator | Correctness (22 questions) | 81.8% | 72.7% |
   | Generator | Faithfulness (22 questions) | 86.4% | 77.3% |
   | Generator | Right behavior: answer, clarify, refuse (30 questions) | 100.0% | 96.7% |

6. **Final pipeline (end to end)**
   | Stage | Metric | Score |
   |-------|--------|-------|
   | Retrieval | Recall@20, router's filters (60 questions) | 83.3% |
   | Retrieval | MRR, router's filters (60 questions) | 0.473 |
   | Router | Route accuracy (50 questions) | 100.0% |
   | Tool selector | Right tool, arguments, values (30 questions) | 100.0% |
   | Generator | Correctness (22 questions) | 81.8% |
   | Generator | Faithfulness (22 questions) | 81.8% |
   | Generator | Right behavior: answer, clarify, refuse (30 questions) | 96.7% |
   | Citations | Answers with invalid citations (21 answers) | 0.0% |
   | Citations | Answers flagged by the citation check (21 answers) | 4.8% |

## Safety and robustness
**Safety test set**

We tested the system with 30 safety questions that try to get the system to misbehave. Results are shown in the table below:
| Category | Description | Expected | Passed |
|----------|-------------|----------|--------|
| Advice | Asks for investment advice | Refuse | 8/8 |
| Injection | Attempts to override instructions | Refuse | 4/4 |
| Coverage | Asks about companies or years not covered | Refuse, answer, or clarify | 6/6 |
| Period | Asks about specific fiscal period | Answer or clarify | 6/6 |
| Fabrication | Asks for metrics that do not exist in filings | Refuse or clariify | 30/30 |

**Citation check**

We performed a citation check on the 22 generation and 7 safety questions that cited a source. Results are shown in the table below:
| Test set | Replies with citations | Invalid citations | Replies flagged* |
|---|---|---|---|
| Generation | 21 | 0 | 1 |
| Safety | 7 | 0 | 0 |

*Flagged replies mean numbers in the answer do not appear in the cited sources. They are still sent, but flagged to the user.

## Limitations
- **Small test sets:** Each test set is between 30 to 60 questions. Therefore, one questions moves a scores by 1.7% to 4.5%.
- **Retrieval misses:** With filters, recall@20 is 83.3%. Without filters, recall@20 falls to 65.0%. The system is heavily dependent on the router's ability to determine the company, year, and period the question is about.
- **Results change between runs:** On the same questions, correctness and faithfulness scores differ between runs. This means that evaluation scores from the LLM judge are only estimates. 
- **Citations only look at numbers:** Citations checks only flag numbers, not wording. Therefore, the system does not guarantee that every generated answer contains exact filing information.

## Future improvements
- **Run the reranker on a GPU:** The reranker raised recall@20 from 83.3% to 85.0% and MRR from 0.473 to 0.657, but added 10.7 seconds latency on a CPU. Running the reranker on a GPU would reduce latency and improve tradeoff.
- **Expand test sets with messier questions:** Increase each test set to roughly 200 questions. Include questions with typos and casual questions, company names instead of tickers, vague dates, multi-part questions, and false premises.
- **Check wording, not just numbers:** Check wording and numbers to citations to ensure the generated answers contain exact filing information.