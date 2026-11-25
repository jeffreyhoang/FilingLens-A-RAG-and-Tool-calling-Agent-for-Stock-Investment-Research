"""Reranker: reorder a query's search results with a cross-encoder.

- Tested in the eval only; the app doesn't use it.

Run with:
    uv run python -m app.retrieval.rerank [--id r001] [--n 50]
"""

import argparse
import time
from functools import cache

import numpy as np

from app.schemas import Chunk

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEVICE = "cpu"  # Railway has no GPU
MAX_LENGTH = 512  # tokens per (query, chunk) pair; a 1000-character chunk fits
BATCH = 16  # pairs per forward pass
TOP_N = 50  # candidates from search
TOP_M = 20  # chunks kept after reranking


@cache
def cross_encoder(model_id: str):
    from sentence_transformers import CrossEncoder  # eval group only

    return CrossEncoder(model_id, device=DEVICE, max_length=MAX_LENGTH)


def rerank(
    query: str,
    chunks: list[Chunk],
    m: int = TOP_M,
    model_id: str = RERANK_MODEL,
) -> list[tuple[Chunk, float]]:
    """The top m chunks by cross-encoder score, best first."""
    if not chunks:
        return []
    scores = cross_encoder(model_id).predict(
        [(query, c.text) for c in chunks], batch_size=BATCH, show_progress_bar=False
    )
    top = np.argsort(-scores, kind="stable")[:m]
    return [(chunks[i], float(scores[i])) for i in top]


def main() -> None:
    from app.eval.sets import load_set  # imported here so search can import this file
    from app.retrieval.embed import CHUNKS_DIR
    from app.retrieval.search import Index

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id", default="r001", help="retrieval-set question")
    parser.add_argument("--n", type=int, default=TOP_N, help="candidates to rerank")
    args = parser.parse_args()

    item = next(i for i in load_set("retrieval") if i.id == args.id)
    index = Index("openai-large-768", CHUNKS_DIR.with_name("chunks-1000"), half=True)
    hits = index.search(item.query, args.n, item.filters)
    chunks = [c for c, _ in hits]

    cross_encoder(RERANK_MODEL)  # load first, so the timer skips it
    t0 = time.perf_counter()
    reranked = rerank(item.query, chunks)
    ms = (time.perf_counter() - t0) * 1000

    print(f"{item.id}: {item.query}\nfilters: {item.filters}\n")
    print("plain search:")
    for c, score in hits[:5]:
        print(f"  {score:7.4f}  {c.chunk_id}")
    print("reranked:")
    for c, score in reranked[:5]:
        print(f"  {score:7.4f}  {c.chunk_id}  (was #{chunks.index(c) + 1})")
    print(f"\n{len(chunks)} chunks reranked in {ms:.0f} ms")


if __name__ == "__main__":
    main()
