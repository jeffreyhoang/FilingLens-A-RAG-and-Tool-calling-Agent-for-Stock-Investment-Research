"""Evidence: gather each generation-set question's sources once, so every generator
model sees the same ones.

Run with:
    uv run python -m app.eval.evidence [--check]
"""

import argparse
import json

from app.eval.run import RESULTS_DIR, covers
from app.eval.sets import GenerationItem, MetricRef, load_set
from app.pipeline import gather, search_index
from app.schemas import Chunk, Metric

OUT = RESULTS_DIR / "contexts/generation.jsonl"
# The route each label implies: quotes only, metric values only, or both
EXPECTED_ROUTE = {
    (True, False): "narrative",
    (False, True): "metric",
    (True, True): "hybrid",
}


def build() -> None:
    """Gather the sources for every question; write one row per question."""
    index = search_index()
    rows = []
    for item in load_set("generation"):
        row = {"id": item.id, **gather(item.query, index)}
        rows.append(row)
        print(f"{item.id} {row['route']:12} chunks {len(row['chunks']):2}  "
              f"tool {row['tool']['tool'] if row['tool'] else '-'}")  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} questions -> {OUT}")


def has_value(rows: list[Metric], ref: MetricRef) -> bool:
    return any(
        (m.ticker, m.metric, m.fiscal_year, m.fiscal_period, m.value)
        == (ref.ticker, ref.metric, ref.fiscal_year, ref.fiscal_period, ref.value)
        for m in rows
    )


def coverage(items: list[GenerationItem], rows: dict[str, dict]) -> dict:
    """For answer questions: labeled quotes found in the chunks and labeled metric
    values found in the tool rows. For all questions: the route against the one
    the labels imply (refuse -> out_of_scope, clarify -> ambiguous)."""
    quotes = values = quotes_found = values_found = 0
    misses, route_ok, wrong_route = [], 0, []
    for item in items:
        row = rows[item.id]
        if item.behavior == "answer":
            expected = EXPECTED_ROUTE[(bool(item.evidence), bool(item.metrics))]
        else:
            expected = {"refuse": "out_of_scope", "clarify": "ambiguous"}[item.behavior]
        if row["route"] == expected:
            route_ok += 1
        else:
            wrong_route.append(f"{item.id}: {expected} -> {row['route']}")

        chunks = [Chunk.model_validate(c) for c in row["chunks"]]
        tool_rows = (row["tool"] or {}).get("rows", [])
        metrics = [Metric.model_validate(m) for m in tool_rows]
        missed_quotes = [
            f"quote from {ev.accession_no} {ev.section}"
            for ev in item.evidence
            if not any(covers(c, ev) for c in chunks)
        ]
        missed_values = [
            f"{m.ticker} {m.metric} {m.fiscal_period}{m.fiscal_year}"
            for m in item.metrics
            if not has_value(metrics, m)
        ]
        quotes += len(item.evidence)
        values += len(item.metrics)
        quotes_found += len(item.evidence) - len(missed_quotes)
        values_found += len(item.metrics) - len(missed_values)
        if missed_quotes or missed_values:
            misses.append({"id": item.id, "missing": missed_quotes + missed_values})
    answer_items = [i for i in items if i.behavior == "answer"]
    return {
        "questions": len(items),
        "route_matches_labels": f"{route_ok}/{len(items)}",
        "wrong_route": wrong_route,
        "answer_questions_with_all_sources": (
            f"{len(answer_items) - len(misses)}/{len(answer_items)}"
        ),
        "quotes_found": f"{quotes_found}/{quotes}",
        "metric_values_found": f"{values_found}/{values}",
        "misses": misses,
    }


def check() -> None:
    rows = {r["id"]: r for r in map(json.loads, OUT.open()) if r}
    report = coverage(load_set("generation"), rows)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="coverage only")
    args = parser.parse_args()
    if not args.check:
        build()
    check()


if __name__ == "__main__":
    main()
