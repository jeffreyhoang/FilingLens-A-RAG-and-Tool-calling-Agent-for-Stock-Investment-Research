"""Spot-check: grade a sample of answers by hand and compare with the judge.

- The sheet is blind: model names and judge scores are hidden.

Run with:
    uv run python -m app.eval.spotcheck --write
    uv run python -m app.eval.spotcheck --score
"""

import argparse
import csv
import json
import random

from app.eval.judge import JUDGE_DIR, JUDGE_METRICS, PRED_DIR
from app.eval.sets import ROOT, load_set

SPOT_DIR = ROOT / "eval/results/spotcheck"
SAMPLE = 10
SEED = 11
MODELS = ("gpt-5.4-mini", "claude-haiku-4-5")
FIELDS = ["id", "answer", "correctness", "faithfulness", "note"]

GUIDE = """\
# Generation spot-check

Grade each answer 0 or 1 in `grades.csv` (same rules the judge uses):

- **correctness:** 1 if the answer states the same key facts as the reference
  answer (same figures, rounding to 3 significant digits is fine, same reasons).
  Extra correct detail is fine. 0 if a key fact is missing, wrong, or contradicted.
- **faithfulness:** 1 if every factual claim is stated in, or directly computed
  from, the sources it cites. 0 if any claim isn't supported, even if it's true.

Answers A and B come from the two models in a random order per question.
"""


def load_jsonl(path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.open()) if r}


def write() -> None:
    from app.eval.evidence import OUT as EVIDENCE
    from app.steps.generator import sources

    grades = SPOT_DIR / "grades.csv"
    filled = grades.exists() and any(
        r["correctness"] or r["faithfulness"] for r in csv.DictReader(grades.open())
    )
    if filled:
        raise SystemExit(f"{grades} already has grades; move it before rewriting")

    items = {i.id: i for i in load_set("generation") if i.behavior == "answer"}
    rng = random.Random(SEED)
    ids = sorted(rng.sample(sorted(items), SAMPLE))
    evidence = load_jsonl(EVIDENCE)
    preds = {m: load_jsonl(PRED_DIR / f"{m}-generation.jsonl") for m in MODELS}

    key, rows, parts = {}, [], [GUIDE]
    for qid in ids:
        item = items[qid]
        order = list(MODELS)
        rng.shuffle(order)
        key[qid] = dict(zip("AB", order, strict=True))
        by_n = {s.n: s for s in sources(evidence[qid])}
        parts += [
            f"\n---\n\n## {qid}: {item.query}\n",
            f"**Reference answer:** {item.reference_answer}\n",
        ]
        for label, model in key[qid].items():
            pred = preds[model][qid]
            cited = [by_n[c["n"]] for c in pred["citations"]]
            parts += [
                f"### Answer {label} ({pred['behavior']})\n",
                f"{pred['answer']}\n",
                f"<details><summary>Cited sources ({len(cited)} of {len(by_n)})"
                "</summary>\n",
            ]
            parts += [f"**[{s.n}] {s.label}**\n\n{s.text}\n" for s in cited]
            parts.append("</details>\n")
            rows.append({"id": qid, "answer": label})

    SPOT_DIR.mkdir(parents=True, exist_ok=True)
    (SPOT_DIR / "sheet.md").write_text("\n".join(parts))
    with grades.open("w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (SPOT_DIR / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    print(f"{len(rows)} answers, questions {', '.join(ids)} -> {SPOT_DIR}")


def score() -> None:
    key = json.loads((SPOT_DIR / "key.json").read_text())
    judged = {m: load_jsonl(JUDGE_DIR / f"{m}-generation.jsonl") for m in MODELS}
    tally = {m: {name: [0, 0] for name in JUDGE_METRICS} for m in MODELS}
    yours = {m: {name: 0 for name in JUDGE_METRICS} for m in MODELS}
    disagreements, ungraded = [], []
    for row in csv.DictReader((SPOT_DIR / "grades.csv").open()):
        model = key[row["id"]][row["answer"]]
        for name in JUDGE_METRICS:
            mine = row[name].strip()
            if mine not in ("0", "1"):
                ungraded.append(f"{row['id']} {row['answer']} {name}")
                continue
            judge = judged[model][row["id"]][name]
            yours[model][name] += int(mine)
            tally[model][name][0] += int(mine) == judge
            tally[model][name][1] += 1
            if int(mine) != judge:
                disagreements.append({
                    "id": row["id"], "model": model, "metric": name,
                    "you": int(mine), "judge": judge,
                    "judge_reason": judged[model][row["id"]][f"{name}_reason"],
                    "note": row["note"],
                })  # fmt: skip

    def rate(pairs: list[list[int]]) -> str:
        agree, total = map(sum, zip(*pairs, strict=True))
        return f"{agree}/{total} ({100 * agree / total:.0f}%)" if total else "-"

    report = {
        "agreement": rate([t for m in MODELS for t in tally[m].values()]),
        "per_metric": {
            name: rate([tally[m][name] for m in MODELS]) for name in JUDGE_METRICS
        },
        "per_model": {m: rate(list(tally[m].values())) for m in MODELS},
        "your_scores": {
            m: {name: f"{yours[m][name]}/{tally[m][name][1]}" for name in JUDGE_METRICS}
            for m in MODELS
        },
        "disagreements": disagreements,
        "ungraded": ungraded,
    }
    (SPOT_DIR / "agreement.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="write the sheet")
    group.add_argument("--score", action="store_true", help="compare with the judge")
    args = parser.parse_args()
    write() if args.write else score()


if __name__ == "__main__":
    main()
