"""Judge: an LLM scores each answer 0 or 1 for correctness and faithfulness.

- Refuse and clarify items are scored by behavior only.
- --checks tests the judge: reference answers should score ~100%, swapped ones 0.

Run with:
    uv run python -m app.eval.judge --predictions FILE [--evidence FILE]
    uv run python -m app.eval.judge --checks
"""

import argparse
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.eval.sets import ROOT

JUDGE_MODEL = "gpt-5.4"  # larger than both candidates; judges every answer
MAX_TOKENS = 2048
JUDGE_DIR = ROOT / "eval/results/judgments"
PRED_DIR = ROOT / "eval/results/predictions"
CHECK_SHIFT = 11  # the wrong-answer check pairs each question with one 11 items on

CORRECTNESS = """\
You grade answers to questions about SEC filings.

Question: {query}
Reference answer: {reference_answer}
Answer to grade: {answer}

Score 1 if the answer states the same key facts as the reference answer: the same
figures (rounding to 3 significant digits is fine, and units must match) and the
same reasons or descriptions. Extra correct detail is fine. Score 0 if any key fact
is missing, wrong, or contradicted.

Reply with JSON only: {{"score": 0 or 1, "reason": "<one sentence>"}}"""

FAITHFULNESS = """\
You check whether an answer is supported by its sources.

Question: {query}
Sources (filing passages and metric values the system retrieved):
{sources}
Answer to check: {answer}

Score 1 if every factual claim in the answer is stated in, or directly computed
from, the sources. Score 0 if any claim is not supported by the sources, even if it
is true.

Reply with JSON only: {{"score": 0 or 1, "reason": "<one sentence>"}}"""

JUDGE_METRICS = {"correctness": CORRECTNESS, "faithfulness": FAITHFULNESS}


class Judgment(BaseModel):
    score: Literal[0, 1]
    reason: str


def judge(text: str, model: str = JUDGE_MODEL) -> tuple[Judgment, dict]:
    """One judge call on a filled-in metric prompt."""
    from app.retrieval.embed import openai_client

    t0 = time.perf_counter()
    resp = openai_client().responses.parse(
        model=model,
        input=text,
        text_format=Judgment,
        reasoning={"effort": "medium"},  # grading needs care; a few calls only
        max_output_tokens=MAX_TOKENS,
    )
    if resp.output_parsed is None:
        raise RuntimeError(f"{model} returned no parsed judgment")
    stats = {
        "ms": round((time.perf_counter() - t0) * 1000),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    return resp.output_parsed, stats


def sources_text(row: dict) -> str:
    """An evidence row's sources, numbered and labeled as the generator saw them."""
    from app.steps.generator import sources

    return "\n\n".join(f"[{s.n}] {s.label}\n{s.text}" for s in sources(row)) or "none"


def judge_file(
    path: Path, model: str = JUDGE_MODEL, evidence_path: Path | None = None
) -> Path:
    """Judge every answer item in a predictions file; write one row per item.
    An item with no prediction or an empty answer scores 0 without a call."""
    from app.eval.evidence import OUT as EVIDENCE
    from app.eval.sets import load_predictions, load_set

    evidence_path = evidence_path or EVIDENCE
    evidence = {r["id"]: r for r in map(json.loads, evidence_path.open())}
    preds = load_predictions(path)
    rows = []
    for item in load_set("generation"):
        if item.behavior != "answer":
            continue
        answer = (preds.get(item.id) and preds[item.id].answer) or ""
        row = {"id": item.id, "ms": 0, "input_tokens": 0, "output_tokens": 0}
        fields = {
            "query": item.query,
            "reference_answer": item.reference_answer,
            "sources": sources_text(evidence[item.id]),
            "answer": answer,
        }
        for name, template in JUDGE_METRICS.items():
            if not answer.strip():
                row[name], row[f"{name}_reason"] = 0, "no answer"
                continue
            out, stats = judge(template.format(**fields), model)
            row[name], row[f"{name}_reason"] = out.score, out.reason
            for k in ("ms", "input_tokens", "output_tokens"):
                row[k] += stats[k]
        rows.append(row)
        scores = f"correct {row['correctness']} faithful {row['faithfulness']}"
        print(f"  {item.id} {scores}")
    out_path = JUDGE_DIR / f"{path.stem}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(rows)
    print(
        f"{path.stem}: {n} answer items judged by {model} -> {out_path}\n"
        f"  correctness {sum(r['correctness'] for r in rows)}/{n}, "
        f"faithfulness {sum(r['faithfulness'] for r in rows)}/{n}, "
        f"{sum(r['input_tokens'] for r in rows)} input and "
        f"{sum(r['output_tokens'] for r in rows)} output tokens"
    )
    return out_path


def write_checks() -> list[Path]:
    """Two predictions files for checking the judge: the reference answers, and
    each answer item given another answer item's reference answer."""
    from app.eval.sets import load_set

    answers = [i for i in load_set("generation") if i.behavior == "answer"]
    swapped = answers[CHECK_SHIFT:] + answers[:CHECK_SHIFT]
    files = {
        "judge-check-oracle": [(i, i.reference_answer) for i in answers],
        "judge-check-wrong": [
            (i, s.reference_answer) for i, s in zip(answers, swapped, strict=True)
        ],
    }
    paths = []
    for name, pairs in files.items():
        path = PRED_DIR / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps({"id": i.id, "behavior": "answer", "answer": a}) + "\n"
                for i, a in pairs
            )
        )
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions", type=Path, help="predictions file to judge")
    group.add_argument("--checks", action="store_true", help="check the judge")
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--evidence", type=Path, help="sources for faithfulness")
    args = parser.parse_args()

    for path in write_checks() if args.checks else [args.predictions]:
        judge_file(path, args.model, args.evidence)


if __name__ == "__main__":
    main()
