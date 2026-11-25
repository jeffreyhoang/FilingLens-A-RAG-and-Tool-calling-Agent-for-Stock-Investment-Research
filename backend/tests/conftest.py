"""Fakes for every model call and the search index, so tests make no API calls and
load no data files."""

import time

import numpy as np
import pytest

from app import pipeline
from app.logger import RunLogger
from app.schemas import Chunk, Metric
from app.steps import tools
from app.steps.router import RouterOutput, SearchFilters

DELAY = 0.2  # seconds each fake search and tool call takes

CHUNK = Chunk(
    ticker="AAPL", cik="0000320193", form_type="10-K", fiscal_year=2024,
    fiscal_period="FY", period_end="2024-09-28", filing_date="2024-11-01",
    accession_no="0000320193-24-000123", section="Item 7", section_title="MD&A",
    text="Total net sales were $391,035 million.",
    source_url="https://www.sec.gov/Archives/edgar/data/320193/doc.htm",
    chunk_id="0000320193-24-000123:Item 7:0", position=0, char_start=0, char_end=38,
)  # fmt: skip

# The metrics table, as app.steps.tools.metrics() would read it from Supabase
REVENUE = Metric(
    ticker="AAPL", fiscal_year=2024, fiscal_period="FY", metric="revenue",
    value="391035000000", unit="USD", period_start="2023-10-01",
    period_end="2024-09-28",
    xbrl_tag="RevenueFromContractWithCustomerExcludingAssessedTax",
    accession_no="0000320193-24-000123", filed="2024-11-01",
)  # fmt: skip
METRICS = {(m.ticker, m.metric, m.fiscal_year, m.fiscal_period): m for m in [REVENUE]}

# A question's first word picks its fake route
ROUTES = {
    "narrative": "narrative",
    "metric": "metric",
    "hybrid": "hybrid",
    "vague": "ambiguous",
    "advice": "out_of_scope",
}
STATS = {"ms": 1, "input_tokens": 1, "output_tokens": 1}


class FakeIndex:
    def embed_query(self, queries):
        time.sleep(DELAY)
        return np.zeros((len(queries), 3)), 5

    def prep(self, vectors):
        return vectors

    def rank(self, vector, k, filters):
        return [(CHUNK, 0.9)]


class Calls:
    """What each fake received, to check what reached later steps."""

    def __init__(self) -> None:
        self.routed: list[str] = []
        self.generated: list[dict] = []
        self.rewrites = 0


@pytest.fixture
def fakes(monkeypatch, tmp_path) -> Calls:
    calls = Calls()

    def route(question, model):
        calls.routed.append(question)
        filters = SearchFilters(ticker="AAPL", fiscal_year=2024, fiscal_period=None)
        return RouterOutput(route=ROUTES[question.split()[0]], filters=filters), STATS

    def select_tool(question):
        time.sleep(DELAY)
        args = {"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024,
                "fiscal_period": "FY"}  # fmt: skip
        return "get_metric", args, STATS

    def generate(row, model):
        calls.generated.append(row)
        cites = []
        if row["chunks"]:
            cites.append({"n": 1, "kind": "filing", "ticker": "AAPL",
                          "accession_no": CHUNK.accession_no, "section": "Item 7",
                          "chunk_id": CHUNK.chunk_id,
                          "source_url": CHUNK.source_url})  # fmt: skip
        if row["tool"]:
            cites.append({"n": len(row["chunks"]) + 1, "kind": "metric",
                          "ticker": "AAPL",
                          "accession_no": row["tool"]["rows"][0]["accession_no"],
                          "metric": "revenue", "fiscal_year": 2024,
                          "fiscal_period": "FY"})  # fmt: skip
        behavior = {"vague": "clarify", "advice": "refuse"}.get(
            row["query"].split()[0], "answer"
        )
        return {"behavior": behavior, "answer": "Revenue was $391 billion [1].",
                "citations": cites, "invalid_citations": [], "sources": len(cites),
                **STATS}  # fmt: skip

    def standalone(question, history):
        calls.rewrites += 1
        return "metric Apple revenue fiscal 2022", STATS

    monkeypatch.setattr(pipeline, "route", route)
    monkeypatch.setattr(pipeline, "select_tool", select_tool)
    monkeypatch.setattr(pipeline, "generate", generate)
    monkeypatch.setattr(pipeline, "standalone", standalone)
    monkeypatch.setattr(pipeline, "search_index", lambda: FakeIndex())
    monkeypatch.setattr(tools, "metrics", lambda: METRICS)
    monkeypatch.setattr(pipeline, "RunLogger", lambda: RunLogger(log_dir=tmp_path))
    return calls
