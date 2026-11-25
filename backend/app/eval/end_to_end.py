"""End-to-end: run a test set through the full pipeline, as /chat does.

- Writes predictions, the sources each answer used, and time, token and cost stats.

Run with:
    uv run python -m app.eval.end_to_end [--set generation|safety] [--stats]
"""

import argparse
import json
import math

from app import pipeline
from app.eval.evidence import OUT as SAVED_EVIDENCE
from app.eval.run import RESULTS_DIR
from app.eval.sets import load_set

SET_NAMES = ("generation", "safety")


def paths(set_name: str):
    """The predictions and contexts files for one set's end-to-end run."""
    name = f"pipeline-{set_name}"
    return (
        RESULTS_DIR / f"predictions/{name}.jsonl",
        RESULTS_DIR / f"contexts/{name}.jsonl",
    )


PRED, CONTEXTS = paths("generation")
STATS = RESULTS_DIR / "end-to-end-stats.json"

# $ per million tokens, list prices (gpt-5.4-mini from third-party price trackers)
LLM_PRICE = (0.75, 4.50)  # gpt-5.4-mini input, output: router, tools, generator
EMBED_PRICE = 0.13  # text-embedding-3-large
STEPS = ("route", "search", "tool", "generate", "total")
LLM_STEPS = ("router", "tool", "generate")  # keys under an evidence row's "stats"


def load(path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def source_periods(row: dict) -> list[dict]:
    """Each source's kind, ticker and period: the filing chunks, then metric rows."""
    keys = ("ticker", "fiscal_year", "fiscal_period")
    return [
        *({"kind": "filing", **{k: c[k] for k in keys}} for c in row["chunks"]),
        *(
            {"kind": "metric", **{k: m[k] for k in keys}}
            for m in (row["tool"] or {}).get("rows", [])
        ),
    ]


def run_set(set_name: str = "generation") -> None:
    pred_path, contexts_path = paths(set_name)
    index = pipeline.search_index()
    preds, rows = [], []
    for item in load_set(set_name):
        r, row = pipeline.run(item.query, index=index)
        preds.append({
            "id": item.id, "behavior": r.behavior, "answer": r.answer,
            "citations": [c.model_dump() for c in r.citations],
            "invalid_citations": r.invalid_citations,
            "unsupported": [u.model_dump() for u in r.unsupported],
            "sources": source_periods(row), "run_id": r.run_id,
            "route": r.route, "ms": r.ms,
        })  # fmt: skip
        rows.append({"id": item.id, **row})
        flags = f" {len(r.unsupported)} flagged" if r.unsupported else ""
        print(f"  {item.id} {r.route:12} {r.behavior:7} {r.ms['total']:6} ms{flags}")
    for path, data in ((pred_path, preds), (contexts_path, rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(d) + "\n" for d in data))
    print(f"{len(preds)} questions -> {pred_path}\n  sources -> {contexts_path}")


def p95(values: list[float]) -> int:
    ranked = sorted(values)
    return round(ranked[math.ceil(0.95 * len(ranked)) - 1])


def timing(preds: list[dict]) -> dict:
    out = {}
    for step in STEPS:
        values = [p["ms"][step] for p in preds if step in p["ms"]]
        if values:
            out[step] = {
                "questions": len(values),
                "mean_ms": round(sum(values) / len(values)),
                "p95_ms": p95(values),
                "max_ms": max(values),
            }
    return out


def cost(row: dict) -> dict:
    stats = row["stats"]
    tokens_in = sum(stats[s]["input_tokens"] for s in LLM_STEPS if s in stats)
    tokens_out = sum(stats[s]["output_tokens"] for s in LLM_STEPS if s in stats)
    embed = stats.get("search", {}).get("embed_tokens", 0)
    usd = (tokens_in * LLM_PRICE[0] + tokens_out * LLM_PRICE[1]) / 1e6
    usd += embed * EMBED_PRICE / 1e6
    return {"input": tokens_in, "output": tokens_out, "embed": embed, "usd": usd}


def stats() -> None:
    preds, rows = load(PRED), {r["id"]: r for r in load(CONTEXTS)}
    saved = {r["id"]: r["route"] for r in load(SAVED_EVIDENCE)}
    sourced = [p for p in preds if p["route"] not in ("ambiguous", "out_of_scope")]
    declined = [p for p in preds if p not in sourced]
    costs = [cost(rows[p["id"]]) for p in preds]
    n = len(costs)
    report = {
        "questions": n,
        "time_all": timing(preds),
        "total_with_sources": timing(sourced).get("total"),
        "total_clarify_or_refuse": timing(declined).get("total"),
        "cost_per_question": {
            "mean_usd": round(sum(c["usd"] for c in costs) / n, 5),
            "max_usd": round(max(c["usd"] for c in costs), 5),
            "mean_input_tokens": round(sum(c["input"] for c in costs) / n),
            "mean_output_tokens": round(sum(c["output"] for c in costs) / n),
            "mean_embed_tokens": round(sum(c["embed"] for c in costs) / n),
        },
        "total_usd": round(sum(c["usd"] for c in costs), 4),
        "routes_changed_from_saved_evidence": [
            f"{p['id']}: {saved.get(p['id'])} -> {p['route']}"
            for p in preds
            if saved.get(p["id"]) != p["route"]
        ],
        "invalid_citations": sum(len(p["invalid_citations"]) for p in preds),
        # None for runs saved before the citation check existed
        "answers_with_unsupported_figures": (
            sum(bool(p["unsupported"]) for p in preds)
            if all("unsupported" in p for p in preds)
            else None
        ),
    }
    STATS.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {STATS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=SET_NAMES, default="generation", dest="name")
    parser.add_argument("--stats", action="store_true", help="stats only, no calls")
    args = parser.parse_args()
    if args.name == "safety":
        if args.stats:
            parser.error("--stats is for the generation set only")
        run_set("safety")
        return
    if not args.stats:
        run_set()
    stats()


if __name__ == "__main__":
    main()
