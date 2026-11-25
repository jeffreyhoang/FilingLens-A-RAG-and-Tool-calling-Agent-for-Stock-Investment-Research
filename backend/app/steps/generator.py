"""Generator: one LLM call writes the answer, clarifying question or refusal, citing
numbered sources.

- Sources are filing chunks, then metric rows, numbered [1], [2], ...
- Code maps each citation back to its filing or row and flags citations that match
  no source.

Run with:
    uv run python -m app.steps.generator [--model M] ["question" ...]
    uv run python -m app.steps.generator --set [--model M]
"""

import argparse
import json
import re
import time
from datetime import date
from decimal import Decimal

from pydantic import BaseModel

from app.eval.sets import Behavior
from app.schemas import Chunk, Metric
from app.steps.router import COMPANIES, PRED_DIR, anthropic_client

MODELS = ("gpt-5.4-mini", "claude-haiku-4-5")
MAX_TOKENS = 2048

# Hand-written checks covering answer, clarify and refuse; none are in eval/sets/
# or in the router's and tool selector's prompt examples.
SAMPLES = [
    "What does Adobe say about generative AI in its fiscal 2024 annual report?",
    "What was Oracle's operating cash flow in fiscal 2022?",
    "What was Alphabet's operating income in fiscal 2023, and what drove the change?",
    "How much cash did they return to shareholders last year?",
    "Which of these companies will grow fastest next year?",
]

METRIC_NAMES = {
    "revenue": "revenue",
    "net_income": "net income",
    "operating_income": "operating income",
    "operating_cash_flow": "operating cash flow",
    "rnd_expense": "research and development expense",
    "total_assets": "total assets",
    "eps_diluted": "diluted earnings per share",
}

PROMPT = """\
You write the replies for a research assistant that answers questions from the \
10-K and 10-Q filings of these 10 companies, fiscal 2016 to 2026:
{companies}

Each message has a question, the route a router chose for it, and numbered \
sources: passages from the filings, and metric values taken from the filings' \
financial data.

Behavior:
- Route ambiguous: behavior clarify. Ask one short question for what is missing \
(the company, the period, or which figure). Don't answer.
- Route out_of_scope: behavior refuse. In one or two sentences, say what you \
can't do (investment advice, predictions, stock prices, or companies and years \
not covered) and what you can help with instead.
- Any other route: behavior answer, using the sources only. If the sources don't \
contain the answer, behavior refuse and say the filings provided don't cover it. \
Don't guess.

Answer rules:
- Use only facts stated in the sources, never outside knowledge, even if true.
- After each sentence that states a fact, cite its sources by number in \
brackets, like [2] or [1][4]. Cite only numbers listed under Sources.
- Give every number with its unit and fiscal period ("Apple's fiscal 2024", \
"Q2 fiscal 2023"). Round dollar amounts to 3 significant digits in billions or \
millions ("$391 billion") unless the question needs more.
- Simple arithmetic on sourced numbers (a difference, a growth rate) is fine; \
say which numbers it uses.
- A metric value that measures something other than what was asked (total \
assets when the question asks about debt) is not an answer. Use the passages \
instead, or say the filings provided don't cover it.
- Fiscal years are each company's own (Microsoft's fiscal 2024 ended in June \
2024).
- Be brief: one to four sentences, or a short list for several periods or \
companies. No investment advice or predictions.

Format example (made-up figures): "Revenue rose 8% to $12.3 billion in fiscal \
2021 [3], driven mainly by higher subscription sales [1]."

Today's date: {today}."""


class GeneratorOutput(BaseModel):
    behavior: Behavior
    answer: str  # with inline citations like [2]


class Source(BaseModel):
    n: int
    label: str  # what the model sees above the text
    text: str
    cite: dict  # what a citation maps back to


def period_name(fiscal_year: int, fiscal_period: str) -> str:
    return f"fiscal {fiscal_year}" if fiscal_period == "FY" else (
        f"{fiscal_period} fiscal {fiscal_year}"
    )  # fmt: skip


def money(value: Decimal, unit: str) -> str:
    if unit == "USD/shares":
        return f"${value:,.2f} per share"
    sign, v = ("-" if value < 0 else ""), abs(value)
    scaled = (
        f"${v / 10**9:,.2f} billion" if v >= 10**9 else f"${v / 10**6:,.1f} million"
    )
    return f"{sign}${v:,.0f} ({sign}{scaled})"


def sources(row: dict) -> list[Source]:
    """Number an evidence row's chunks, then its metric rows."""
    out = []
    for c in map(Chunk.model_validate, row["chunks"]):
        period = period_name(c.fiscal_year, c.fiscal_period)
        out.append(
            Source(
                n=len(out) + 1,
                label=(
                    f"{c.ticker} {c.form_type}, {period} (period ending "
                    f"{c.period_end}), {c.section}: {c.section_title}"
                ),
                text=c.text,
                cite={
                    "kind": "filing",
                    "ticker": c.ticker,
                    "accession_no": c.accession_no,
                    "section": c.section,
                    "chunk_id": c.chunk_id,
                    "source_url": c.source_url,
                },
            )
        )
    for m in map(Metric.model_validate, (row["tool"] or {}).get("rows", [])):
        name = METRIC_NAMES[m.metric]
        out.append(
            Source(
                n=len(out) + 1,
                label=(
                    f"{m.ticker} {name}, {period_name(m.fiscal_year, m.fiscal_period)} "
                    f"(period ending {m.period_end})"
                ),
                text=f"{name}: {money(m.value, m.unit)}",
                cite={
                    "kind": "metric",
                    "ticker": m.ticker,
                    "accession_no": m.accession_no,
                    "metric": m.metric,
                    "fiscal_year": m.fiscal_year,
                    "fiscal_period": m.fiscal_period,
                },
            )
        )
    return out


def message(row: dict, srcs: list[Source]) -> str:
    """The user message: question, route, numbered sources, and tool notes."""
    parts = [f"Question: {row['query']}", f"Route: {row['route']}", "", "Sources:"]
    parts += [f"[{s.n}] {s.label}\n{s.text}\n" for s in srcs] or ["none"]
    tool = row["tool"]
    if tool and tool.get("note"):
        parts.append(f"Metric lookup note: {tool['note']}")
    if tool and tool.get("missing"):
        gaps = ", ".join(
            f"{g['ticker']} {g['fiscal_period']} {g['fiscal_year'] or 'any year'}"
            for g in tool["missing"]
        )
        parts.append(f"Metric lookup found no data for: {gaps}")
    return "\n".join(parts)


def prompt(today: date) -> str:
    companies = "\n".join(
        f"{c['ticker']}  {c['name']}  fiscal year ends {c['fiscal_year_end']}"
        for c in COMPANIES
    )
    return PROMPT.format(companies=companies, today=today.isoformat())


def call(system: str, user: str, model: str) -> tuple[GeneratorOutput, object]:
    if model.startswith("gpt-"):
        from app.retrieval.embed import openai_client

        resp = openai_client().responses.parse(
            model=model,
            instructions=system,
            input=user,
            text_format=GeneratorOutput,
            reasoning={"effort": "low"},  # matches the router and tool selector
            max_output_tokens=MAX_TOKENS,
        )
        return resp.output_parsed, resp.usage
    if model.startswith("claude-"):
        resp = anthropic_client().messages.parse(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=GeneratorOutput,
        )
        return resp.parsed_output, resp.usage
    raise ValueError(f"unknown model {model!r}")


def generate(row: dict, model: str, today: date | None = None) -> dict:
    """One LLM call for one evidence row. Returns the behavior, the answer, the
    citations (each cited source's `cite`), cited numbers that match no source,
    and {ms, input_tokens, output_tokens}."""
    srcs = sources(row)
    t0 = time.perf_counter()
    out, usage = call(prompt(today or date.today()), message(row, srcs), model)
    if out is None:
        raise RuntimeError(f"{model} returned no parsed output for {row['query']!r}")
    by_n = {s.n: s for s in srcs}
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", out.answer)})
    return {
        "behavior": out.behavior,
        "answer": out.answer,
        "citations": [{"n": n, **by_n[n].cite} for n in cited if n in by_n],
        "invalid_citations": [n for n in cited if n not in by_n],
        "sources": len(srcs),
        "ms": round((time.perf_counter() - t0) * 1000),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


def run_set(model: str) -> None:
    """Answer every generation-set question from the saved evidence; write one row
    per question."""
    from app.eval.evidence import OUT as EVIDENCE

    rows = [json.loads(line) for line in EVIDENCE.open() if line.strip()]
    preds = []
    for row in rows:
        out = generate(row, model)
        preds.append({"id": row["id"], **out})
        print(f"  {row['id']} {out['behavior']:7} {out['ms']:5} ms")
    path = PRED_DIR / f"{model}-generation.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(p) + "\n" for p in preds))
    n = len(preds)
    print(
        f"{model}: {n} questions -> {path}\n"
        f"  mean {sum(p['ms'] for p in preds) / n:.0f} ms per call, "
        f"{sum(p['input_tokens'] for p in preds)} input and "
        f"{sum(p['output_tokens'] for p in preds)} output tokens in total, "
        f"{sum(len(p['invalid_citations']) for p in preds)} invalid citations"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, help="default: both")
    parser.add_argument("--set", action="store_true", help="run the generation set")
    parser.add_argument("questions", nargs="*", help="default: the samples")
    args = parser.parse_args()

    models = [args.model] if args.model else MODELS
    if args.set:
        for model in models:
            run_set(model)
        return

    from app.pipeline import gather, search_index

    index = search_index()
    for q in args.questions or SAMPLES:
        row = gather(q, index)
        tool = row["tool"]["tool"] if row["tool"] else "-"
        print(
            f"\n{q}\n  route {row['route']}, {len(row['chunks'])} chunks, tool {tool}"
        )
        for model in models:
            out = generate(row, model)
            print(f"  == {model}: {out['behavior']}\n  {out['answer']}")
            for c in out["citations"]:
                where = c.get("section") or f"{c['metric']} {c['fiscal_period']}"
                print(f"    [{c['n']}] {c['ticker']} {c['accession_no']} {where}")
            if out["invalid_citations"]:
                print(f"    invalid citations: {out['invalid_citations']}")
            stats = {k: out[k] for k in ("ms", "input_tokens", "output_tokens")}
            print(f"    {stats}")


if __name__ == "__main__":
    main()
