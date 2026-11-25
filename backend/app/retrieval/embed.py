"""Embedder: chunks -> one normalized vector per chunk, saved per company.

- Models: bge-base, gte-modernbert (local), openai-small, openai-large, and their
  768-dimension versions openai-small-768, openai-large-768.
- Writes data/processed/embeddings/<model>/<chunks dir>/ and skips companies already
  done unless --force.

Run with:
    uv run python -m app.retrieval.embed --model M [--chunks DIR] [--tickers T ...]
        [--force]
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # repo root
CHUNKS_DIR = ROOT / "data/processed/chunks"  # chunker output, 2000/200
EMB_DIR = ROOT / "data/processed/embeddings"  # gitignored

LOCAL_BATCH = 32  # chunks per forward pass
OPENAI_BATCH = 256  # chunks per request (about 130k tokens at 2000 characters)


@dataclass(frozen=True)
class Model:
    id: str  # Hugging Face or OpenAI model name
    dims: int
    local: bool = True
    query_prefix: str = ""  # added to queries only, as the model card says
    api_dims: bool = False  # ask the OpenAI API for `dims` numbers per vector


MODELS = {
    "bge-base": Model(
        "BAAI/bge-base-en-v1.5",
        768,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "gte-modernbert": Model("Alibaba-NLP/gte-modernbert-base", 768),
    "openai-small": Model("text-embedding-3-small", 1536, local=False),
    "openai-large": Model("text-embedding-3-large", 3072, local=False),
    "openai-small-768": Model("text-embedding-3-small", 768, False, api_dims=True),
    "openai-large-768": Model("text-embedding-3-large", 768, False, api_dims=True),
}

# texts -> (vectors, tokens used); tokens is 0 for local models
EmbedFn = Callable[[list[str]], tuple[np.ndarray, int]]


@cache
def local_model(model_id: str):
    from sentence_transformers import SentenceTransformer  # eval group only

    return SentenceTransformer(model_id)  # picks the M2 GPU (mps) when available


@cache
def openai_client():
    from openai import OpenAI

    from app.config import get_settings

    return OpenAI(api_key=get_settings().openai_api_key, max_retries=10)


def unit(vectors: np.ndarray) -> np.ndarray:
    """Scale rows to length 1, so a dot product is cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embedder(name: str, query: bool = False) -> EmbedFn:
    """The embedding function for a model; `query` adds the model's query prefix."""
    model = MODELS[name]
    prefix = model.query_prefix if query else ""

    def embed(texts: list[str]) -> tuple[np.ndarray, int]:
        texts = [prefix + t for t in texts]
        if model.local:
            vectors = local_model(model.id).encode(
                texts, batch_size=LOCAL_BATCH, show_progress_bar=len(texts) > 1
            )
            return unit(vectors), 0
        rows, tokens = [], 0
        extra = {"dimensions": model.dims} if model.api_dims else {}
        for i in range(0, len(texts), OPENAI_BATCH):
            resp = openai_client().embeddings.create(
                model=model.id, input=texts[i : i + OPENAI_BATCH], **extra
            )
            rows += [d.embedding for d in resp.data]
            tokens += resp.usage.total_tokens
        return unit(np.array(rows)), tokens

    return embed


def embed_dir(
    chunks_dir: Path,
    out_dir: Path,
    embed: EmbedFn,
    dims: int,
    tickers: list[str] | None = None,
    force: bool = False,
) -> None:
    """Embed each company's chunks file in `chunks_dir` into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(chunks_dir.glob("*.jsonl")):
        ticker = path.stem
        if tickers and ticker not in tickers:
            continue
        if (out_dir / f"{ticker}.npy").exists() and not force:
            print(f"{ticker:5} already embedded, skipping")
            continue
        rows = [json.loads(line) for line in path.open()]
        t0 = time.perf_counter()
        vectors, tokens = embed([r["text"] for r in rows])
        seconds = round(time.perf_counter() - t0, 1)
        if vectors.shape != (len(rows), dims):
            want = (len(rows), dims)
            raise ValueError(f"{ticker}: got {vectors.shape}, want {want}")
        # ids and stats first, vectors last: a .npy file means the company is done
        (out_dir / f"{ticker}.ids.txt").write_text(
            "".join(r["chunk_id"] + "\n" for r in rows)
        )
        stats = {"chunks": len(rows), "seconds": seconds, "tokens": tokens}
        (out_dir / f"{ticker}.stats.json").write_text(json.dumps(stats) + "\n")
        np.save(out_dir / f"{ticker}.npy", vectors)
        print(f"{ticker:5} {len(rows):6} chunks in {seconds:7.1f}s, {tokens} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--chunks", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--tickers", nargs="*", help="default: all companies")
    parser.add_argument("--force", action="store_true", help="re-embed done ones")
    args = parser.parse_args()

    chunks_dir = args.chunks.resolve()
    out_dir = EMB_DIR / args.model / chunks_dir.name
    print(f"{args.model} ({MODELS[args.model].id}): {chunks_dir} -> {out_dir}")
    embed_dir(
        chunks_dir,
        out_dir,
        embedder(args.model),
        MODELS[args.model].dims,
        args.tickers,
        args.force,
    )


if __name__ == "__main__":
    main()
