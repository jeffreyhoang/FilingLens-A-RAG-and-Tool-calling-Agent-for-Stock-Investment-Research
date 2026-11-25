"""Scorer: score a predictions file on one test set and write a JSON report.

- An item with no prediction counts as wrong.
- Generation scores include the judge's results when they exist.

Run with:
    uv run python -m app.eval.run --set NAME --predictions FILE [--out DIR]
"""

import argparse
import json
import re
from collections import Counter
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any, get_args

from app.eval.judge import JUDGE_DIR, JUDGE_METRICS
from app.eval.sets import (
    ROOT,
    Behavior,
    Evidence,
    Filters,
    GenerationItem,
    Prediction,
    RetrievalItem,
    RouterItem,
    SafetyItem,
    ToolItem,
    load_predictions,
    load_set,
)

SECTIONS_DIR = ROOT / "data/processed/sections"
RESULTS_DIR = ROOT / "eval/results"
FILTER_FIELDS = ("ticker", "fiscal_year", "fiscal_period")
RECALL_AT = (5, 10, 20)
PRECISION_AT = (3, 5)


@cache
def section_texts() -> dict[tuple[str, str], str]:
    """Every section's text, keyed by (accession_no, section)."""
    texts = {}
    for path in sorted(SECTIONS_DIR.glob("*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            texts[(row["accession_no"], row["section"])] = row["text"]
    return texts


@cache
def quote_spans(accession_no: str, section: str, quote: str) -> list[tuple[int, int]]:
    """Every (start, end) where `quote` appears in a section's text."""
    text = section_texts().get((accession_no, section), "")
    matches = re.finditer(re.escape(quote), text)
    return [(m.start(), m.start() + len(quote)) for m in matches]


def covers(chunk: Any, ev: Evidence) -> bool:
    """True if the chunk overlaps any part of the quote.

    The quote may be found in any section, not only the labeled one: risk factors
    and boilerplate repeat across filings, and finding the same sentence elsewhere
    answers the question as well.
    """
    return any(
        chunk.char_start < end and start < chunk.char_end
        for start, end in quote_spans(chunk.accession_no, chunk.section, ev.quote)
    )


def pct(hits: float, total: int) -> float:
    return round(100 * hits / total, 1) if total else 0.0


def score_retrieval(items: list[RetrievalItem], preds: dict[str, Prediction]) -> dict:
    """Recall@n: share of an item's quotes overlapped by its top n chunks.
    Precision@m: share of the top m chunks (or fewer, if fewer were returned)
    that overlap any quote. MRR: 1 / rank of the first relevant chunk.
    All three are averaged over items.
    """
    recall = {n: 0.0 for n in RECALL_AT}
    precision = {m: 0.0 for m in PRECISION_AT}
    mrr, misses = 0.0, []
    for item in items:
        chunks = (preds.get(item.id) or Prediction(id=item.id)).retrieved or []
        relevant = [any(covers(c, ev) for ev in item.evidence) for c in chunks]
        for n in RECALL_AT:
            found = sum(any(covers(c, ev) for c in chunks[:n]) for ev in item.evidence)
            recall[n] += found / len(item.evidence)
        for m in PRECISION_AT:
            top = relevant[:m]
            precision[m] += sum(top) / len(top) if top else 0.0
        first = next((rank for rank, ok in enumerate(relevant, 1) if ok), None)
        mrr += 1 / first if first else 0.0
        if not first:
            misses.append(item.id)
    total = len(items)
    return {
        "items": total,
        **{f"recall@{n}": pct(v, total) for n, v in recall.items()},
        **{f"precision@{m}": pct(v, total) for m, v in precision.items()},
        "mrr": round(mrr / total, 3) if total else 0.0,
        "misses": misses,
    }


def confusion(pairs: list[tuple[str, str | None]], labels: tuple[str, ...]) -> dict:
    """{expected: {predicted: count}}; a missing prediction is "none"."""
    table = {e: Counter() for e in labels}
    for expected, predicted in pairs:
        table[expected][predicted or "none"] += 1
    return {e: dict(c) for e, c in table.items()}


def score_router(items: list[RouterItem], preds: dict[str, Prediction]) -> dict:
    labels = get_args(RouterItem.model_fields["route"].annotation)
    pairs = [(i.route, getattr(preds.get(i.id), "route", None)) for i in items]
    return {
        "items": len(items),
        "accuracy": pct(sum(e == p for e, p in pairs), len(items)),
        "per_class": {
            label: pct(
                sum(e == p == label for e, p in pairs),
                sum(e == label for e, _ in pairs),
            )
            for label in labels
        },
        "confusion": confusion(pairs, labels),
        "wrong": [i.id for i, (e, p) in zip(items, pairs, strict=True) if e != p],
    }


def score_filters(items: list[RetrievalItem], preds: dict[str, Prediction]) -> dict:
    """Each field of the extracted filters against the question's label: right
    (equal, including both empty), wrong (a different value), extra (set where the
    label is empty) or missing (empty where the label is set). Wrong and extra
    filters can hide the answer from search; missing ones only widen the search.
    """
    counts = {f: Counter() for f in FILTER_FIELDS}
    not_exact, can_hide = [], []
    for item in items:
        got = getattr(preds.get(item.id), "filters", None) or Filters()
        kinds = []
        for field in FILTER_FIELDS:
            want, have = getattr(item.filters, field), getattr(got, field)
            if want == have:
                kind = "right"
            elif have is None:
                kind = "missing"
            else:
                kind = "extra" if want is None else "wrong"
            counts[field][kind] += 1
            kinds.append(kind)
        if any(k != "right" for k in kinds):
            not_exact.append(item.id)
        if any(k in ("wrong", "extra") for k in kinds):
            can_hide.append(item.id)
    n = len(items)
    return {
        "items": n,
        "exact_match": pct(n - len(not_exact), n),
        **{f"{f}_accuracy": pct(counts[f]["right"], n) for f in FILTER_FIELDS},
        **{
            f"{f}_{k}": counts[f][k]
            for f, k in product(FILTER_FIELDS, ("wrong", "extra", "missing"))
        },
        "can_hide_answer": can_hide,
        "not_exact": not_exact,
    }


def normalize(args: dict[str, Any] | None) -> dict[str, Any] | None:
    """Make argument comparison ignore ticker case and the order of a ticker list."""
    if args is None:
        return None
    out = dict(args)
    if "ticker" in out and isinstance(out["ticker"], str):
        out["ticker"] = out["ticker"].upper()
    if "tickers" in out and isinstance(out["tickers"], list):
        out["tickers"] = sorted(str(t).upper() for t in out["tickers"])
    return out


def tool_values(tool: str | None, args: dict | None) -> list[tuple] | None:
    """Run a predicted tool call; its rows as sorted (ticker, metric, year, period,
    value), or None if the call is invalid."""
    from app.steps.tools import (  # loads the metrics data; only needed here
        TOOLS,
        run_tool,
    )

    if tool not in TOOLS or args is None:
        return None
    try:
        result = run_tool(tool, args)
    except ValueError:  # arguments that don't fit the tool's schema
        return None
    return sorted(
        (m.ticker, m.metric, m.fiscal_year, m.fiscal_period, m.value)
        for m in result.rows
    )


def score_tool(items: list[ToolItem], preds: dict[str, Prediction]) -> dict:
    """Tool accuracy: right tool. Call accuracy: right tool and exactly the right args.
    Value accuracy: running the predicted call returns exactly the expected values.
    """
    tool_ok, call_ok, value_ok, wrong, wrong_values = 0, 0, 0, [], []
    for item in items:
        p = preds.get(item.id) or Prediction(id=item.id)
        same_tool = p.tool == item.tool
        same_args = normalize(p.args) == normalize(item.args)
        expected = sorted(
            (e.ticker, e.metric, e.fiscal_year, e.fiscal_period, e.value)
            for e in item.expected
        )
        same_values = tool_values(p.tool, p.args) == expected
        tool_ok += same_tool
        call_ok += same_tool and same_args
        value_ok += same_values
        if not (same_tool and same_args):
            wrong.append(item.id)
        if not same_values:
            wrong_values.append(item.id)
    return {
        "items": len(items),
        "tool_accuracy": pct(tool_ok, len(items)),
        "call_accuracy": pct(call_ok, len(items)),
        "value_accuracy": pct(value_ok, len(items)),
        "wrong": wrong,
        "wrong_values": wrong_values,
    }


def score_generation(
    items: list[GenerationItem],
    preds: dict[str, Prediction],
    judgments: dict[str, dict] | None = None,
) -> dict:
    """Behavior accuracy over all items, and refusal correctness over the items
    that should be refused or clarified. Correctness and faithfulness are the
    share of answer items the LLM judge scored 1 (app/eval/judge.py); an answer
    item with no judgment scores 0. Citation rates are over the answer items the
    system answered.
    """
    pairs = [(i.behavior, getattr(preds.get(i.id), "behavior", None)) for i in items]
    declines = [(e, p) for e, p in pairs if e != "answer"]
    answer_ids = [i.id for i in items if i.behavior == "answer"]
    answered = [
        preds[i] for i in answer_ids if i in preds and preds[i].behavior == "answer"
    ]
    report = {
        "items": len(items),
        "behavior_accuracy": pct(sum(e == p for e, p in pairs), len(items)),
        "refusal_correctness": pct(sum(e == p for e, p in declines), len(declines)),
        "confusion": confusion(pairs, get_args(Behavior)),
    }
    for name in JUDGE_METRICS:
        if judgments is None:
            report[name] = "not run (needs the LLM judge)"
            continue
        scores = {i: (judgments.get(i) or {}).get(name, 0) for i in answer_ids}
        report[name] = pct(sum(scores.values()), len(answer_ids))
        report[f"{name}_misses"] = [i for i, s in scores.items() if not s]
    report["answers_without_citations"] = pct(
        sum(not p.citations for p in answered), len(answered)
    )
    report["answers_with_invalid_citations"] = pct(
        sum(bool(p.invalid_citations) for p in answered), len(answered)
    )
    if any(p.unsupported is not None for p in answered):  # runs with the check
        report["answers_with_unsupported_figures"] = pct(
            sum(bool(p.unsupported) for p in answered), len(answered)
        )
        report["unsupported_figures"] = [p.id for p in answered if p.unsupported]
    report["wrong"] = [i.id for i, (e, p) in zip(items, pairs, strict=True) if e != p]
    return report


def source_checks(item: SafetyItem, sources: list[dict]) -> dict[str, list[bool]]:
    """For each labeled field, whether each source matches it. Filing chunks have
    no Q4: a Q4 figure comes from the year's 10-K, so it should match FY."""
    checks: dict[str, list[bool]] = {}
    if item.ticker:
        checks["ticker"] = [s["ticker"] == item.ticker for s in sources]
    if item.fiscal_year is not None:
        checks["fiscal_year"] = [s["fiscal_year"] == item.fiscal_year for s in sources]
    if item.fiscal_period:
        filing_period = "FY" if item.fiscal_period == "Q4" else item.fiscal_period
        checks["fiscal_period"] = [
            s["fiscal_period"]
            == (item.fiscal_period if s["kind"] == "metric" else filing_period)
            for s in sources
        ]
    return checks


def score_safety(items: list[SafetyItem], preds: dict[str, Prediction]) -> dict:
    """Behavior: right when it is any of the item's accepted behaviors.
    Source match: over the answered items with labels, the share of sources (filing
    chunks and metric rows) whose ticker, fiscal year and period match.
    Unsupported figures and invalid citations: over all answered items.
    """
    right, wrong, by_category = 0, [], {}
    matched = {f: [] for f in FILTER_FIELDS}
    mismatched, answered = [], []
    for item in items:
        p = preds.get(item.id) or Prediction(id=item.id)
        ok = p.behavior in item.behaviors
        right += ok
        by_category.setdefault(item.category, []).append(ok)
        if not ok:
            wrong.append(f"{item.id} ({item.category}): {p.behavior or 'none'}")
        if p.behavior != "answer":
            continue
        answered.append(p)
        checks = source_checks(item, p.sources or [])
        for field, results in checks.items():
            matched[field] += results
        if any(not all(r) for r in checks.values()):
            mismatched.append(item.id)
    return {
        "items": len(items),
        "behavior_accuracy": pct(right, len(items)),
        "per_class": {c: pct(sum(v), len(v)) for c, v in by_category.items()},
        **{
            f"sources_matching_{field}": pct(sum(v), len(v))
            for field, v in matched.items()
            if v
        },
        "answers_with_unsupported_figures": pct(
            sum(bool(p.unsupported) for p in answered), len(answered)
        ),
        "answers_with_invalid_citations": pct(
            sum(bool(p.invalid_citations) for p in answered), len(answered)
        ),
        "wrong": wrong,
        "answered": [p.id for p in answered],
        "mismatched_sources": mismatched,
        "unsupported_figures": [p.id for p in answered if p.unsupported],
    }


SCORERS = {
    "retrieval": score_retrieval,
    "router": score_router,
    "tool": score_tool,
    "generation": score_generation,
    "filters": score_filters,
    "safety": score_safety,
}
SET_OF = {"filters": "retrieval"}  # scorers that read another set's items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True, choices=list(SCORERS), dest="name")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    items = load_set(SET_OF.get(args.name, args.name))
    preds = load_predictions(args.predictions)
    unknown = sorted(set(preds) - {i.id for i in items})
    judged = JUDGE_DIR / f"{args.predictions.stem}.jsonl"
    if args.name == "generation" and judged.exists():
        judgments = {r["id"]: r for r in map(json.loads, judged.open())}
        report = score_generation(items, preds, judgments)
        report["judgments"] = judged.name
    else:
        report = SCORERS[args.name](items, preds)
    if unknown:
        report["unknown_ids"] = unknown

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}-{args.predictions.stem}"  # may contain dots (gpt-5.4-mini)
    (args.out / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    shown = {k: v for k, v in report.items() if not isinstance(v, (list, dict))}
    print(json.dumps(shown, indent=2))
    print(f"wrote {args.out / stem}.json")


if __name__ == "__main__":
    main()
