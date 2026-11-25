"""Tool selector: one LLM call picks a metric tool and its arguments.

Run with:
    uv run python -m app.steps.tool_selector ["question" ...]
    uv run python -m app.steps.tool_selector --set
"""

import argparse
import json
import time
from datetime import date

from openai.lib._pydantic import to_strict_json_schema

from app.eval.sets import load_set
from app.retrieval.embed import openai_client
from app.steps.router import COMPANIES, PRED_DIR
from app.steps.tools import TOOLS, metrics, run_tool

MODEL = "gpt-5.4-mini"  # the router's model
MAX_TOKENS = 1024

DESCRIPTIONS = {
    "get_metric": "One company's metric for one fiscal period.",
    "compare_companies": "One metric for one fiscal period across two or more "
    "companies.",
    "get_metric_series": "One company's metric for each fiscal year in a range.",
}

# Hand-written checks, one or two per tool; none are in eval/sets/ or PROMPT.
SAMPLES = [
    "What was Microsoft's operating profit in the fourth quarter of fiscal 2022?",
    "What's NVIDIA's revenue?",
    "Did Apple or Amazon generate more cash from operations in fiscal 2019?",
    "Show Adobe's R&D spending over the last three years.",
]

PROMPT = """\
You pick one tool to look up reported financial metrics from the 10-K and 10-Q \
filings of the 10 companies listed below, and fill in its arguments from the \
question. Always call exactly one tool.

Tools:
- get_metric: one company, one metric, one period.
- compare_companies: the same metric and period for two or more companies \
("compare", "which had more", "rank").
- get_metric_series: one company's metric over a range of fiscal years ("each \
year", "from 2019 to 2023", "trend", "how has it changed").

Metrics:
- revenue: revenue, sales.
- net_income: net income, profit, earnings.
- operating_income: operating income, operating profit.
- operating_cash_flow: cash from operations, operating cash flow.
- rnd_expense: research and development (R&D) spending.
- total_assets: total assets (a year-end or quarter-end balance).
- eps_diluted: diluted earnings per share, EPS.

Periods:
- fiscal_year is the company's own fiscal year, named by the calendar year it \
ends in (Microsoft's fiscal 2024 ended in June 2024). A year in the question \
means that fiscal year.
- fiscal_period: Q1, Q2, Q3 or Q4 for a named quarter ("fourth quarter" is Q4), \
and FY for a full year. A year with no quarter means FY.
- No year given: set fiscal_year to null (the tool uses the most recent year \
with data).
- Relative ranges ("the last five years"): count back from the company's latest \
full fiscal year below, including it.

Companies (ticker, name, fiscal year end, latest full fiscal year with data):
{companies}

Examples:
- "How much did Oracle spend on research and development in the third quarter \
of fiscal 2019?" -> get_metric(ticker ORCL, metric rnd_expense, fiscal_year \
2019, fiscal_period Q3)
- "What is Tesla's most recent earnings per share?" -> get_metric(ticker TSLA, \
metric eps_diluted, fiscal_year null, fiscal_period FY)
- "Rank Netflix, Adobe and Amazon by sales in the second quarter of fiscal \
2021." -> compare_companies(tickers [NFLX, ADBE, AMZN], metric revenue, \
fiscal_year 2021, fiscal_period Q2)
- "Chart Alphabet's profit over the last four fiscal years." -> \
get_metric_series(ticker GOOGL, metric net_income, start_year {googl_start}, \
end_year {googl_end}, fiscal_period FY)

Today's date: {today}."""


def latest_years() -> dict[str, int]:
    """Each company's latest fiscal year with full-year revenue in the data."""
    latest: dict[str, int] = {}
    for ticker, metric, year, period in metrics():
        if metric == "revenue" and period == "FY":
            latest[ticker] = max(year, latest.get(ticker, year))
    return latest


def prompt(today: date) -> str:
    latest = latest_years()
    companies = "\n".join(
        f"{c['ticker']}  {c['name']}  {c['fiscal_year_end']}  FY{latest[c['ticker']]}"
        for c in COMPANIES
    )
    return PROMPT.format(
        companies=companies,
        googl_start=latest["GOOGL"] - 3,
        googl_end=latest["GOOGL"],
        today=today.isoformat(),
    )


def tool_schemas() -> list[dict]:
    """The 3 tools as strict OpenAI function schemas."""
    return [
        {
            "type": "function",
            "name": name,
            "description": DESCRIPTIONS[name],
            "parameters": to_strict_json_schema(model),
            "strict": True,
        }
        for name, (model, _) in TOOLS.items()
    ]


def select_tool(
    question: str, model: str = MODEL, today: date | None = None
) -> tuple[str, dict, dict]:
    """One LLM call. Returns the tool name, its arguments, and
    {ms, input_tokens, output_tokens}."""
    t0 = time.perf_counter()
    resp = openai_client().responses.create(
        model=model,
        instructions=prompt(today or date.today()),
        input=question,
        tools=tool_schemas(),
        tool_choice="required",
        parallel_tool_calls=False,
        reasoning={"effort": "low"},  # a small task; keeps each call quick
        max_output_tokens=MAX_TOKENS,
    )
    calls = [item for item in resp.output if item.type == "function_call"]
    if len(calls) != 1:
        raise RuntimeError(f"expected one tool call for {question!r}, got {calls}")
    args = json.loads(calls[0].arguments)
    stats = {
        "ms": round((time.perf_counter() - t0) * 1000),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    return calls[0].name, args, stats


def run_set(model: str = MODEL) -> None:
    """Select a tool for every tool-set question; write one row per question."""
    rows = []
    for item in load_set("tool"):
        name, args, stats = select_tool(item.query, model)
        rows.append({"id": item.id, "tool": name, "args": args, **stats})
    path = PRED_DIR / f"{model}-tool.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(rows)
    print(
        f"{model}: {n} questions -> {path}\n"
        f"  mean {sum(r['ms'] for r in rows) / n:.0f} ms per call, "
        f"{sum(r['input_tokens'] for r in rows)} input and "
        f"{sum(r['output_tokens'] for r in rows)} output tokens in total"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", action="store_true", help="run the tool test set")
    parser.add_argument("questions", nargs="*", help="default: the samples")
    args = parser.parse_args()

    if args.set:
        run_set()
        return
    for q in args.questions or SAMPLES:
        name, tool_args, stats = select_tool(q)
        result = run_tool(name, tool_args)
        rows = [
            f"{m.ticker} {m.fiscal_period}{m.fiscal_year} {m.value}"
            for m in result.rows
        ]
        print(f"{q}\n  {name}({json.dumps(tool_args)})\n  rows: {rows}")
        if result.missing or result.note:
            print(f"  missing: {result.missing}  note: {result.note}")
        print(f"  {stats}")


if __name__ == "__main__":
    main()
