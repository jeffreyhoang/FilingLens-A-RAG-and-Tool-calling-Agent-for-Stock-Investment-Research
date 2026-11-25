"""Citation check: every number an answer states must be in the sources it cites.

- A figure is supported if a cited source has it at the answer's precision
  ($391 billion matches $391,035 million), or one arithmetic step on two source
  numbers gives it (difference, percent change, ratio).
- A sentence without a citation is checked against all the answer's cited sources.
- Skipped: counts under 10, citation markers, form names, quarters, section numbers.
- No LLM call. It can raise false alarms or miss errors, so it flags, not blocks.

Run with:
    uv run python -m app.steps.citecheck [--predictions FILE] [--contexts FILE]
"""

import argparse
import json
import re
from dataclasses import dataclass
from functools import cache
from itertools import permutations
from math import floor, log10
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

RESULTS_DIR = Path(__file__).resolve().parents[3] / "eval/results"

Kind = Literal["amount", "percent", "multiple", "year"]

SCALE_WORDS = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "t": 1e12,
}  # fmt: skip
# Filings write amounts in dollars, or in tables "in thousands" or "in millions"
TABLE_SCALES = (1.0, 1e3, 1e6)

# Text with digits that isn't a figure; removed before figures are read
NOT_FIGURES = re.compile(
    r"\[\d+(?:\s*,\s*\d+)*\]"  # citation markers
    r"|\b(?:10-K|10-Q|8-K)\b"  # form names
    r"|\bQ[1-4]\b"  # quarters
    r"|\b(?:Items?|Notes?|Part)\s+\d+[A-Z]?\b"  # filing sections
    r"|\b\d{4}-\d{2}-\d{2}\b"  # ISO dates in period labels
    r"|\b\d{4}-\d{2}-\d{6}\b",  # accession numbers
    re.IGNORECASE,
)
FIGURE = re.compile(
    r"(?<![\w.])(?P<cur>\$)?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?"
    r"(?:\s?(?P<unit>%|percent(?:age points?)?\b|times\b|x\b"
    r"|thousand\b|million\b|billion\b|trillion\b|bn\b|mn\b|(?-i:[KMBT])\b))?",
    re.IGNORECASE,
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])|\n+")
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class SourceLike(Protocol):
    """app.steps.generator.Source: what the generator saw as source [n]."""

    n: int
    label: str
    text: str
    cite: dict


class Unsupported(BaseModel):
    """A sentence with figures that its sources don't have."""

    sentence: str  # exactly as it appears in the answer
    numbers: list[str]  # the unsupported figures, as written
    cited: list[int]  # the sources it was checked against


@dataclass(frozen=True)
class Figure:
    text: str  # as written
    value: float  # absolute value, with any scale word applied
    digits: int  # significant digits as written
    kind: Kind


def significant_digits(whole: str, decimals: str | None) -> int:
    """'391' -> 3, '4.4' -> 2, '4,400' -> 2 (trailing zeros of a whole number
    don't count), '0.50' -> 2."""
    digits = whole.replace(",", "")
    if decimals is None:
        digits = digits.rstrip("0")
    else:
        digits += decimals
    return max(len(digits.lstrip("0")), 1)


def figures(text: str, keep_small: bool = False) -> list[Figure]:
    """The figures in a piece of text. Answers skip bare counts under 10 (list
    numbers, "two segments"); sources keep everything."""
    out = []
    for m in FIGURE.finditer(NOT_FIGURES.sub(" ", text)):
        whole, dec = m["int"], m["dec"]
        unit = (m["unit"] or "").lower()
        value = float(whole.replace(",", "") + (f".{dec}" if dec else ""))
        if unit.startswith("percent") or unit == "%":
            kind: Kind = "percent"
        elif unit in ("x", "times"):
            kind = "multiple"
        elif not (m["cur"] or unit or dec) and 1900 <= value <= 2100:
            kind = "year"
        else:
            kind = "amount"
            value *= SCALE_WORDS.get(unit, 1.0)
            bare = not (m["cur"] or unit or dec)
            if bare and value < 10 and not keep_small:
                continue
        out.append(Figure(m[0].strip(), value, significant_digits(whole, dec), kind))
    return out


def near(fig: Figure, x: float, scales: tuple[float, ...] = (1.0,)) -> bool:
    """True if x, times one of the scales, rounds to the figure as written."""
    if fig.value == 0:
        return x == 0
    half = 0.5 * 10 ** (floor(log10(fig.value)) - fig.digits + 1)
    return any(abs(abs(x) * s - fig.value) <= half * (1 + 1e-9) for s in scales)


@dataclass
class Evidence:
    """What a set of sources supports: their numbers, one-step arithmetic on
    them, and their raw text (for years)."""

    values: list[float]
    differences: list[float]
    changes: list[float]  # percent changes
    ratios: list[float]
    text: str


def evidence(srcs: list[SourceLike]) -> Evidence:
    groups = [
        [f.value for f in figures(f"{s.label}\n{s.text}", keep_small=True)
         if f.kind != "year"]
        for s in srcs
    ]  # fmt: skip
    metric_values = [v for s, g in zip(srcs, groups, strict=True)
                     if s.cite.get("kind") == "metric" for v in g]  # fmt: skip
    differences, changes, ratios = [], [], []
    for group in [*groups, metric_values] if len(metric_values) > 1 else groups:
        for a, b in permutations(set(group), 2):
            differences.append(abs(b - a))
            if a:
                changes.append(abs(100 * (b - a) / a))
                ratios.append(abs(b / a))
    return Evidence(
        values=[v for g in groups for v in g],
        differences=differences,
        changes=changes,
        ratios=ratios,
        text="\n".join(f"{s.label}\n{s.text}" for s in srcs),
    )


def supported(fig: Figure, ev: Evidence) -> bool:
    if fig.kind == "year":
        return str(int(fig.value)) in ev.text
    if fig.kind == "percent":
        return any(near(fig, x) for x in (*ev.values, *ev.changes, *ev.differences))
    if fig.kind == "multiple":
        return any(near(fig, x) for x in (*ev.ratios, *ev.values))
    return any(near(fig, x, TABLE_SCALES) for x in (*ev.values, *ev.differences))


def sentences(text: str) -> list[str]:
    """Split an answer into sentences and list lines, each exactly as written."""
    out, start = [], 0
    for m in SENTENCE_END.finditer(text):
        out.append(text[start : m.start()])
        start = m.end()
    out.append(text[start:])
    return [s.strip() for s in out if s.strip()]


def cited_numbers(text: str) -> list[int]:
    return sorted({int(n) for m in CITATION.findall(text) for n in m.split(",")})


def check(answer: str, srcs: list[SourceLike]) -> list[Unsupported]:
    """The answer's sentences whose figures their cited sources don't support."""
    by_n = {s.n: s for s in srcs}
    answer_cites = [n for n in cited_numbers(answer) if n in by_n] or list(by_n)

    @cache
    def evidence_for(ns: tuple[int, ...]) -> Evidence:
        return evidence([by_n[n] for n in ns])

    flagged = []
    for sentence in sentences(answer):
        figs = figures(sentence)
        if not figs:
            continue
        ns = tuple(n for n in cited_numbers(sentence) if n in by_n) or tuple(
            answer_cites
        )
        ev = evidence_for(ns)
        missing = [f.text for f in figs if not supported(f, ev)]
        if missing:
            flagged.append(
                Unsupported(sentence=sentence, numbers=missing, cited=list(ns))
            )
    return flagged


def main() -> None:
    from app.steps.generator import sources

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--predictions",
        type=Path,
        default=RESULTS_DIR / "predictions/pipeline-generation.jsonl",
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=RESULTS_DIR / "contexts/pipeline-generation.jsonl",
    )
    args = parser.parse_args()

    rows = {r["id"]: r for r in map(json.loads, args.contexts.open())}
    answers, flags = 0, []
    for pred in map(json.loads, args.predictions.open()):
        if pred["behavior"] != "answer" and not pred.get("citations"):
            continue  # as in the pipeline: answers, and declines that cite
        answers += 1
        for u in check(pred["answer"], sources(rows[pred["id"]])):
            flags.append({"id": pred["id"], **u.model_dump()})
            print(
                f"{pred['id']}  {u.numbers}  checked against {u.cited}\n  {u.sentence}"
            )
    flagged = sorted({f["id"] for f in flags})
    report = {
        "predictions": args.predictions.name,
        "checked_replies": answers,
        "flagged_replies": len(flagged),
        "flagged_ids": flagged,
        "flags": flags,
    }
    out = RESULTS_DIR / f"citecheck-{args.predictions.stem}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(flagged)} of {answers} replies flagged -> {out}")


if __name__ == "__main__":
    main()
