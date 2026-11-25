"""Router: one LLM call turns a question into a route and search filters.

Run with:
    uv run python -m app.steps.router [--model M] ["question" ...]
    uv run python -m app.steps.router --set router|retrieval [--model M]
"""

import argparse
import json
import time
from datetime import date
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.eval.sets import Filters, Route, load_set
from app.schemas import ChunkPeriod

ROOT = Path(__file__).resolve().parents[3]  # repo root
COMPANIES = json.loads((ROOT / "data/companies.json").read_text())
PRED_DIR = ROOT / "eval/results/predictions"
TICKERS = tuple(c["ticker"] for c in COMPANIES)
Ticker = Literal[TICKERS]  # the model can only name a covered company

MODELS = ("gpt-5.4-mini", "claude-haiku-4-5")
MAX_TOKENS = 1024  # the output is a short JSON object

# Hand-written, one per route; none are in eval/sets/.
SAMPLES = [
    "How does Oracle describe the risks of expanding its cloud data centers?",
    "What was Cisco's diluted EPS in the second quarter of fiscal 2021?",
    "How much did Adobe's operating income change in fiscal 2020, and why?",
    "Which company had the better margins?",
    "Should I sell my Netflix shares before earnings?",
]

PROMPT = """\
You route questions for FilingLens, a research assistant that answers questions \
from the 10-K and 10-Q filings of the 10 companies listed below, for fiscal years \
2016 to 2026. It gives research, not financial advice.

For each question, return two things.

1. route, one of:
- narrative: answered from the text of the filings, such as risks, strategy, \
competition, products, legal matters or management's explanations.
- metric: asks only for reported numbers (revenue, net income, operating income, \
operating cash flow, R&D expense, total assets or diluted EPS) for one or more \
companies and periods.
- hybrid: asks for one of those numbers and also for an explanation from the \
filing text, such as how much something changed and why.
- ambiguous: about the covered companies and their filings, but it can't be \
answered without more detail, for example when it doesn't say which company \
("they", "the company") or which companies to compare.
- out_of_scope: anything else: investment advice (buy, sell, hold, which stock is \
better to own), share price or other predictions, companies not listed below, and \
topics unrelated to the filings.

2. filters, used to search the filings. Fill a field only when the question \
states it, otherwise leave it null. A wrong filter hides the answer, so when \
unsure, leave it null.
- ticker: the company's ticker from the list below. Null if the question names \
no listed company, or more than one.
- fiscal_year: the fiscal year the question asks about. Companies name a fiscal \
year by the calendar year it ends in (Microsoft's fiscal 2024 ended in June \
2024), and a year in the question means that fiscal year. For relative years \
("last year"), work it out from today's date and the company's fiscal year end.
- fiscal_period: Q1, Q2 or Q3 when the question names that quarter. FY only \
when it says "full year", "annual report", "10-K" or "fourth quarter". Null when \
it names only a year, even for yearly figures or changes ("revenue in fiscal \
2022", "how did income change in 2020"), or names no period.

Companies (ticker, name, fiscal year end):
{companies}

Examples:
- "What does Amazon say about labor shortages in its fulfillment network?" -> \
narrative; ticker AMZN, no year, no period
- "What total assets did Microsoft report in its fiscal 2019 annual report?" \
-> metric; ticker MSFT, fiscal_year 2019, fiscal_period FY
- "What was Tesla's net income in fiscal 2021, and what does management credit \
for it?" -> hybrid; ticker TSLA, fiscal_year 2021, no period
- "What did the company say about its debt?" -> ambiguous; no ticker, no year, \
no period
- "What was Intel's revenue in 2022?" -> out_of_scope (Intel isn't covered); no \
ticker, fiscal_year 2022, no period

Today's date: {today}."""


class SearchFilters(BaseModel):
    ticker: Ticker | None
    fiscal_year: int | None
    fiscal_period: ChunkPeriod | None


class RouterOutput(BaseModel):
    route: Route
    filters: SearchFilters

    def search_filters(self) -> Filters:
        """The filters in the form search uses (`Index.rank`)."""
        return Filters(**self.filters.model_dump())


def prompt(today: date) -> str:
    companies = "\n".join(
        f"{c['ticker']}  {c['name']}  {c['fiscal_year_end']}  {c['notes']}".rstrip()
        for c in COMPANIES
    )
    return PROMPT.format(companies=companies, today=today.isoformat())


@cache
def anthropic_client():
    from anthropic import Anthropic

    from app.config import get_settings

    return Anthropic(api_key=get_settings().anthropic_api_key, max_retries=10)


def route(
    question: str, model: str, today: date | None = None
) -> tuple[RouterOutput, dict]:
    """One LLM call. Returns the parsed output and {ms, input_tokens, output_tokens}."""
    system = prompt(today or date.today())
    t0 = time.perf_counter()
    if model.startswith("gpt-"):
        from app.retrieval.embed import openai_client

        resp = openai_client().responses.parse(
            model=model,
            instructions=system,
            input=question,
            text_format=RouterOutput,
            reasoning={"effort": "low"},  # a small task; keeps each call quick
            max_output_tokens=MAX_TOKENS,
        )
        out, usage = resp.output_parsed, resp.usage
    elif model.startswith("claude-"):
        resp = anthropic_client().messages.parse(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": question}],
            output_format=RouterOutput,
        )
        out, usage = resp.parsed_output, resp.usage
    else:
        raise ValueError(f"unknown model {model!r}")
    if out is None:
        raise RuntimeError(f"{model} returned no parsed output for {question!r}")
    stats = {
        "ms": round((time.perf_counter() - t0) * 1000),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    return out, stats


def run_set(name: str, model: str) -> Path:
    """Route every question in a test set; write one row per question."""
    rows = []
    for item in load_set(name):
        out, stats = route(item.query, model)
        rows.append({"id": item.id, **out.model_dump(), **stats})
    path = PRED_DIR / f"{model}-{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(rows)
    print(
        f"{model}: {n} questions -> {path}\n"
        f"  mean {sum(r['ms'] for r in rows) / n:.0f} ms per call, "
        f"{sum(r['input_tokens'] for r in rows)} input and "
        f"{sum(r['output_tokens'] for r in rows)} output tokens in total"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, help="default: both")
    parser.add_argument("--set", choices=["router", "retrieval"], help="test set")
    parser.add_argument("questions", nargs="*", help="default: the samples")
    args = parser.parse_args()

    models = [args.model] if args.model else MODELS
    if args.set:
        for model in models:
            run_set(args.set, model)
        return
    for model in models:
        print(f"== {model}")
        for q in args.questions or SAMPLES:
            out, stats = route(q, model)
            print(f"{q}\n  {out.route}  {out.filters.model_dump()}")
            print(f"  {stats}")


if __name__ == "__main__":
    main()
