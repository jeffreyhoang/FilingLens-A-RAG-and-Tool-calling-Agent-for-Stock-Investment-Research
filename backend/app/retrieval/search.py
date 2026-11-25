"""Search: rank chunks by cosine similarity to a query, with optional filters (local).

- Used to compare embedding setups on the retrieval test set.
- --dims shortens vectors, --half rounds them to 16-bit (as Supabase stores them).

Run with:
    uv run python -m app.retrieval.search --model M [--chunks DIR] [--filters]
        [--dims N] [--half] [--rerank N] [--router FILE] [--source local|supabase]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.eval.sets import Filters, Prediction, RetrievedChunk, load_set
from app.retrieval.embed import CHUNKS_DIR, EMB_DIR, MODELS, ROOT, embedder
from app.retrieval.rerank import RERANK_MODEL, cross_encoder, rerank
from app.schemas import Chunk

PRED_DIR = ROOT / "eval/results/predictions"
TOP_K = 20
FILTER_FIELDS = ("ticker", "fiscal_year", "fiscal_period", "form_type")


class Index:
    """Every embedded chunk of one model and chunk size, held in memory."""

    def __init__(
        self,
        model: str,
        chunks_dir: Path = CHUNKS_DIR,
        dims: int | None = None,
        half: bool = False,
    ) -> None:
        self.model, self.dims, self.half = model, dims, half
        emb_dir = EMB_DIR / model / chunks_dir.name
        vectors, self.chunks = [], []
        for npy in sorted(emb_dir.glob("*.npy")):
            ids = (emb_dir / f"{npy.stem}.ids.txt").read_text().splitlines()
            rows = [
                Chunk.model_validate_json(line)
                for line in (chunks_dir / f"{npy.stem}.jsonl").open()
            ]
            if [r.chunk_id for r in rows] != ids:
                raise ValueError(f"{npy.stem}: chunks changed since embedding")
            vectors.append(np.load(npy))
            self.chunks += rows
        if not vectors:
            raise FileNotFoundError(f"no embeddings in {emb_dir}")
        self.vectors = self.prep(np.concatenate(vectors))  # (chunks, dims), unit rows
        self.meta = {
            f: np.array([getattr(c, f) for c in self.chunks]) for f in FILTER_FIELDS
        }
        self.embed_query = embedder(model, query=True)

    def prep(self, vectors: np.ndarray) -> np.ndarray:
        """Shorten to `dims` and rescale, then round to 16-bit if `half`."""
        if self.dims:
            if self.dims > vectors.shape[1]:
                raise ValueError(f"--dims {self.dims} > model size {vectors.shape[1]}")
            vectors = vectors[:, : self.dims]
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        if self.half:
            vectors = vectors.astype(np.float16).astype(np.float32)
        return vectors

    def mask(self, filters: Filters | None) -> np.ndarray:
        """True for chunks that match every filter that is set."""
        keep = np.ones(len(self.chunks), dtype=bool)
        for field, value in (filters.model_dump() if filters else {}).items():
            if value is not None:
                keep &= self.meta[field] == value
        return keep

    def rank(
        self, query_vec: np.ndarray, k: int = TOP_K, filters: Filters | None = None
    ) -> list[tuple[Chunk, float]]:
        """Top k chunks by cosine similarity (a dot product of unit vectors)."""
        scores = self.vectors @ query_vec
        scores[~self.mask(filters)] = -np.inf
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > -np.inf]

    def search(
        self, query: str, k: int = TOP_K, filters: Filters | None = None
    ) -> list[tuple[Chunk, float]]:
        vectors, _ = self.embed_query([query])
        return self.rank(self.prep(vectors)[0], k, filters)


def predictions(
    index: Index,  # or app.retrieval.supabase.SupabaseIndex (same interface)
    use_filters: bool,
    k: int = TOP_K,
    rerank_n: int | None = None,
    router: dict[str, dict] | None = None,
) -> tuple[list[Prediction], list[float]]:
    """Search every retrieval-set query. Filters come from each query's labels
    (`use_filters`) or from the router's output (`router`, rows keyed by id).

    Returns the predictions and, with `rerank_n`, the reranking ms per query.
    """
    items = load_set("retrieval")
    if router and (missing := [i.id for i in items if i.id not in router]):
        raise ValueError(f"router output has no row for {missing}")
    queries = [i.query for i in items]
    if rerank_n:
        cross_encoder(RERANK_MODEL)  # load first, so the timer skips it
    query_vecs, _ = index.embed_query(queries)
    query_vecs = index.prep(query_vecs)
    preds, rerank_ms = [], []
    for item, query, vec in zip(items, queries, query_vecs, strict=True):
        if router:
            filters = Filters(**router[item.id]["filters"])
        else:
            filters = item.filters if use_filters else None
        if rerank_n:
            candidates = [c for c, _ in index.rank(vec, rerank_n, filters)]
            t0 = time.perf_counter()
            hits = rerank(query, candidates, k)
            rerank_ms.append((time.perf_counter() - t0) * 1000)
        else:
            hits = index.rank(vec, k, filters)
        retrieved = [
            RetrievedChunk(
                accession_no=c.accession_no,
                section=c.section,
                char_start=c.char_start,
                char_end=c.char_end,
            )
            for c, _ in hits
        ]
        preds.append(Prediction(id=item.id, retrieved=retrieved))
    return preds, rerank_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--chunks", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--filters", action="store_true", help="use query labels")
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--dims", type=int, help="shorten vectors to N numbers")
    parser.add_argument("--half", action="store_true", help="round to 16-bit")
    parser.add_argument("--rerank", type=int, metavar="N", help="rerank top N to k")
    parser.add_argument("--router", type=Path, help="router output on retrieval set")
    parser.add_argument(
        "--source", choices=["local", "supabase"], default="local", help="chunks from"
    )
    args = parser.parse_args()
    if args.router and args.filters:
        parser.error("--router replaces --filters; use one")

    router = None
    if args.router:
        rows = [json.loads(line) for line in args.router.open() if line.strip()]
        router = {r["id"]: r for r in rows}
    if args.source == "supabase":
        from app.retrieval.supabase import CHUNKS_NAME, MODEL, SupabaseIndex

        if args.model != MODEL or not args.half or args.dims:
            parser.error(f"Supabase holds only --model {MODEL} --half (no --dims)")
        index, chunks_name, suffix = SupabaseIndex(), CHUNKS_NAME, "-supabase"
        n_chunks = index.count()
    else:
        chunks_dir = args.chunks.resolve()
        index = Index(args.model, chunks_dir, args.dims, args.half)
        chunks_name, suffix, n_chunks = chunks_dir.name, "", len(index.chunks)
    preds, rerank_ms = predictions(index, args.filters, args.k, args.rerank, router)
    if args.router:
        tag = f"router-{args.router.stem}"
    else:
        tag = "filters" if args.filters else "nofilters"
    size = (f"-d{args.dims}" if args.dims else "") + ("-half" if args.half else "")
    size += f"-rerank{args.rerank}" if args.rerank else ""
    out = PRED_DIR / f"{args.model}-{chunks_name}{size}-{tag}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(p.model_dump_json(exclude_none=True) + "\n" for p in preds))
    print(f"{n_chunks} chunks searched, {len(preds)} queries -> {out}")
    if rerank_ms:
        stats = {
            "candidates": args.rerank,
            "questions": len(rerank_ms),
            "mean_ms": round(float(np.mean(rerank_ms)), 1),
            "p95_ms": round(float(np.percentile(rerank_ms, 95)), 1),
        }
        out.with_suffix(".stats.json").write_text(json.dumps(stats) + "\n")
        print(stats)


if __name__ == "__main__":
    main()
