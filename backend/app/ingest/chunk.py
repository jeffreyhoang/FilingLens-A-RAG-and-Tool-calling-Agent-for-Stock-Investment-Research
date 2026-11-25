"""Chunker: cleaned sections -> overlapping chunks with full metadata.

Run with:
    uv run python -m app.ingest.chunk [--size N] [--overlap N] [--out DIR]
"""

import argparse
import statistics
from collections.abc import Iterator
from pathlib import Path

from app.schemas import Chunk, Section

ROOT = Path(__file__).resolve().parents[3]  # repo root
SECTIONS_DIR = ROOT / "data/processed/sections"  # extractor output
OUT_DIR = ROOT / "data/processed/chunks"  # one <TICKER>.jsonl per company

DEFAULT_SIZE = 2_000  # characters (about 500 tokens)
DEFAULT_OVERLAP = 200  # characters repeated from the end of the previous chunk
BREAKS = ["\n\n", "\n", ". ", " "]  # preferred places to end a chunk, best first


def windows(text: str, size: int, overlap: int) -> Iterator[tuple[int, int]]:
    """Yield (start, end) offsets of overlapping windows over `text`.

    Each window ends at the last paragraph, line, sentence, or word break in its
    second half (or at `size` if there is none). The next window starts `overlap`
    characters before that end, moved forward to the start of a word.
    """
    if not 0 <= overlap < size // 2:
        raise ValueError("overlap must be at least 0 and under half the size")
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            for sep in BREAKS:
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        yield start, end
        if end >= n:
            break
        start = end - overlap
        while start < end and not text[start - 1].isspace():
            start += 1  # don't begin mid-word


def chunk_section(
    section: Section, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    """Split a section into overlapping chunks of at most `size` characters."""
    text = section.text
    chunks = []
    for start, end in windows(text, size, overlap):
        while start < end and text[start].isspace():  # trim, keeping offsets exact
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        position = len(chunks)
        chunks.append(
            Chunk(
                **section.model_dump(exclude={"text"}),
                chunk_id=f"{section.accession_no}:{section.section}:{position}",
                position=position,
                text=text[start:end],
                char_start=start,
                char_end=end,
            )
        )
    return chunks


def chunk_file(path: Path, size: int, overlap: int) -> list[Chunk]:
    """Chunk every section in one company's sections JSONL."""
    sections = [Section.model_validate_json(line) for line in path.open()]
    return [c for s in sections for c in chunk_section(s, size, overlap)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    # e.g. --out ../data/processed/chunks-1000
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths = []
    for path in sorted(SECTIONS_DIR.glob("*.jsonl")):
        chunks = chunk_file(path, args.size, args.overlap)
        out = out_dir / path.name
        out.write_text("".join(c.model_dump_json() + "\n" for c in chunks))
        lengths += [len(c.text) for c in chunks]
        print(f"{path.stem:5} {len(chunks):6} chunks -> {out}")
    q = statistics.quantiles(lengths, n=10)
    print(
        f"Total {len(lengths)} chunks (size {args.size}, overlap {args.overlap}) | "
        f"length min {min(lengths)}, p10 {q[0]:.0f}, median "
        f"{statistics.median(lengths):.0f}, p90 {q[-1]:.0f}, max {max(lengths)}"
    )


if __name__ == "__main__":
    main()
