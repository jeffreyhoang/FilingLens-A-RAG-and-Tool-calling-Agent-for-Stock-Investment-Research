"""Answer one question from the filings.

Steps:
    standalone  rewrite a follow-up to stand on its own (only with chat history)
    route       pick the route and search filters
    search      top 20 Supabase chunks with those filters (narrative, metric, hybrid)
    tool        call a metric tool on the metric rows (metric, hybrid)
    generate    write the answer, clarifying question or refusal, with citations
    check       flag sentences whose figures their cited sources lack (not blocked)

Notes:
    - Search and tool run in parallel. Metric questions also search, because some
      ask for figures no tool covers (debt, purchase obligations, subscribers).
    - Each step logs one line to logs/<YYYY-MM-DD>.jsonl under the response's run id.

Run from backend/ (calls the OpenAI API; no questions runs the generator's SAMPLES):
    uv run python -m app.pipeline ["question" ...]
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Literal

from pydantic import BaseModel

from app.eval.sets import Behavior, Filters, Route
from app.logger import RunLogger
from app.retrieval.supabase import TOP_K, SupabaseIndex
from app.steps.citecheck import Unsupported, check
from app.steps.generator import generate, sources
from app.steps.router import COMPANIES, SearchFilters, route
from app.steps.standalone import Message, standalone
from app.steps.tool_selector import select_tool
from app.steps.tools import ToolResult, run_tool

# The setup chosen in the evaluations: router and generator. The embedding model
# is app.retrieval.supabase.MODEL, the only embeddings loaded into Supabase.
ROUTER_MODEL = "gpt-5.4-mini"
GENERATOR_MODEL = "gpt-5.4-mini"

SEARCH_ROUTES = ("narrative", "metric", "hybrid")
TOOL_ROUTES = ("metric", "hybrid")
CIKS = {c["ticker"]: c["cik"] for c in COMPANIES}


class Citation(BaseModel):
    n: int  # the [n] in the answer
    kind: Literal["filing", "metric"]
    ticker: str
    accession_no: str
    url: str  # the filing document, or the filing's folder on EDGAR for a metric
    section: str | None = None  # filing
    chunk_id: str | None = None  # filing
    metric: str | None = None  # metric
    fiscal_year: int | None = None  # metric
    fiscal_period: str | None = None  # metric


class Response(BaseModel):
    run_id: str
    question: str  # as asked
    standalone_question: str  # what the pipeline answered; same as asked on turn 1
    route: Route
    filters: SearchFilters
    behavior: Behavior
    answer: str  # with inline citations like [2]
    citations: list[Citation]
    invalid_citations: list[int]  # cited numbers that match no source
    unsupported: list[Unsupported]  # sentences with figures their sources lack
    tool: ToolResult | None  # rows and chart hint for a table or chart
    ms: dict[str, int]  # per step and total


@cache
def search_index() -> SupabaseIndex:
    """Search over the chunks in Supabase; connects on first use."""
    return SupabaseIndex()


def filing_url(ticker: str, accession_no: str) -> str:
    """The filing's folder on EDGAR."""
    cik, folder = int(CIKS[ticker]), accession_no.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/"


def search_step(
    question: str, filters: Filters, index: SupabaseIndex, log: RunLogger
) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    with log.step("search", query=question, filters=filters, k=TOP_K) as out:
        vectors, tokens = index.embed_query([question])
        hits = index.rank(index.prep(vectors)[0], TOP_K, filters)
        out["chunk_ids"] = [c.chunk_id for c, _ in hits]
        out["embed_tokens"] = tokens
    chunks = [{**c.model_dump(mode="json"), "score": round(s, 4)} for c, s in hits]
    stats = {"ms": round((time.perf_counter() - t0) * 1000), "embed_tokens": tokens}
    return chunks, stats


def tool_step(question: str, log: RunLogger) -> tuple[dict, dict]:
    with log.step("tool", query=question) as out:
        name, args, stats = select_tool(question)
        result = run_tool(name, args)
        out.update(
            tool=name, args=args, rows=len(result.rows), missing=result.missing,
            note=result.note, **stats,
        )  # fmt: skip
    return result.model_dump(mode="json"), stats


def gather(question: str, index: SupabaseIndex, log: RunLogger | None = None) -> dict:
    """Route one question and gather its sources: the route, the router's filters,
    the chunks, the tool result, and each step's stats."""
    log = log or RunLogger()
    with log.step("route", query=question, model=ROUTER_MODEL) as out:
        routed, stats = route(question, ROUTER_MODEL)
        out.update(route=routed.route, filters=routed.filters, **stats)
    row = {
        "query": question,
        "route": routed.route,
        "filters": routed.filters.model_dump(),
        "chunks": [],
        "tool": None,
        "stats": {"router": stats},
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        search = tool = None
        if routed.route in SEARCH_ROUTES:
            filters = routed.search_filters()
            search = pool.submit(search_step, question, filters, index, log)
        if routed.route in TOOL_ROUTES:
            tool = pool.submit(tool_step, question, log)
        if search:
            row["chunks"], row["stats"]["search"] = search.result()
        if tool:
            row["tool"], row["stats"]["tool"] = tool.result()
    return row


def answer(
    question: str,
    history: list[Message] | None = None,
    index: SupabaseIndex | None = None,
) -> Response:
    """Answer one question, given the earlier messages of the conversation: make it
    stand on its own, gather its sources, then generate the reply."""
    return run(question, history, index)[0]


def run(
    question: str,
    history: list[Message] | None = None,
    index: SupabaseIndex | None = None,
) -> tuple[Response, dict]:
    """As `answer`, and also return the sources the reply was written from (the
    evidence row), with every step's time and tokens under "stats"."""
    log = RunLogger()
    t0 = time.perf_counter()
    asked, history = question, history or []
    standalone_stats = None
    if history:
        with log.step("standalone", query=asked, messages=len(history)) as out:
            question, stats = standalone(asked, history)
            out.update(question=question, **stats)
        standalone_stats = stats
    row = gather(question, index or search_index(), log)
    with log.step("generate", query=question, model=GENERATOR_MODEL) as out:
        gen = generate(row, GENERATOR_MODEL)
        out.update(
            behavior=gen["behavior"],
            answer=gen["answer"],
            citations=[c["n"] for c in gen["citations"]],
            invalid_citations=gen["invalid_citations"],
            sources=gen["sources"],
            ms=gen["ms"],
            input_tokens=gen["input_tokens"],
            output_tokens=gen["output_tokens"],
        )
    unsupported: list[Unsupported] = []
    if gen["behavior"] == "answer" or gen["citations"]:  # a decline can cite too
        with log.step("check", query=question) as out:
            unsupported = check(gen["answer"], sources(row))
            out["unsupported"] = [u.model_dump() for u in unsupported]
    citations = [
        Citation(
            **{k: v for k, v in c.items() if k in Citation.model_fields},
            url=c.get("source_url") or filing_url(c["ticker"], c["accession_no"]),
        )
        for c in gen["citations"]
    ]
    stats = row["stats"]
    stats["generate"] = {k: gen[k] for k in ("ms", "input_tokens", "output_tokens")}
    if standalone_stats:
        stats["standalone"] = standalone_stats
    ms = {
        **({"standalone": standalone_stats["ms"]} if standalone_stats else {}),
        "route": stats["router"]["ms"],
        **{step: stats[step]["ms"] for step in ("search", "tool") if step in stats},
        "generate": gen["ms"],
        "total": round((time.perf_counter() - t0) * 1000),
    }
    response = Response(
        run_id=log.run_id,
        question=asked,
        standalone_question=question,
        route=row["route"],
        filters=SearchFilters(**row["filters"]),
        behavior=gen["behavior"],
        answer=gen["answer"],
        citations=citations,
        invalid_citations=gen["invalid_citations"],
        unsupported=unsupported,
        tool=ToolResult.model_validate(row["tool"]) if row["tool"] else None,
        ms=ms,
    )
    return response, row


def main() -> None:
    from app.steps.generator import SAMPLES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", nargs="*", help="default: the samples")
    args = parser.parse_args()

    index = search_index()
    for q in args.questions or SAMPLES:
        r = answer(q, index=index)
        tool = r.tool.tool if r.tool else "-"
        filters = r.filters.model_dump()
        print(f"\n{q}\n  run {r.run_id}: {r.route} {filters}, tool {tool}")
        print(f"  {r.behavior}: {r.answer}")
        for c in r.citations:
            where = c.section or f"{c.metric} {c.fiscal_period}{c.fiscal_year}"
            print(f"    [{c.n}] {c.ticker} {where} {c.url}")
        for u in r.unsupported:
            print(f"    unsupported {u.numbers}: {u.sentence}")
        print(f"  ms {r.ms}")


if __name__ == "__main__":
    main()
