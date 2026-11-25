"""Oracle: write perfect and empty predictions for a set, to test the scorer.

- Scoring them should give 100% and 0%.

Run with:
    uv run python -m app.eval.oracle --set NAME --out DIR
"""

import argparse
from pathlib import Path

from app.eval.run import quote_spans
from app.eval.sets import (
    SETS,
    GenerationItem,
    Prediction,
    RetrievalItem,
    RetrievedChunk,
    RouterItem,
    ToolItem,
    load_set,
)


def oracle(item: RetrievalItem | RouterItem | ToolItem | GenerationItem) -> Prediction:
    """The prediction a perfect system would make for one item."""
    if isinstance(item, RetrievalItem):  # one chunk exactly covering each quote
        chunks = []
        for ev in item.evidence:
            start, end = quote_spans(ev.accession_no, ev.section, ev.quote)[0]
            chunks.append(
                RetrievedChunk(
                    accession_no=ev.accession_no,
                    section=ev.section,
                    char_start=start,
                    char_end=end,
                )
            )
        return Prediction(id=item.id, retrieved=chunks)
    if isinstance(item, RouterItem):
        return Prediction(id=item.id, route=item.route)
    if isinstance(item, ToolItem):
        return Prediction(id=item.id, tool=item.tool, args=item.args)
    return Prediction(id=item.id, behavior=item.behavior, answer=item.reference_answer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True, choices=list(SETS), dest="name")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    items = load_set(args.name)
    rows = [oracle(item).model_dump_json(exclude_none=True) for item in items]
    (args.out / "oracle.jsonl").write_text("\n".join(rows) + "\n")
    (args.out / "empty.jsonl").write_text("")
    print(f"wrote {len(rows)} oracle rows and an empty file to {args.out}")


if __name__ == "__main__":
    main()
