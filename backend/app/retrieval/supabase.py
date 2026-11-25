"""Supabase search: the app's chunk search, one SQL query per question.

- Filters by ticker, fiscal year, period and form type, then ranks by cosine
  distance. Exact search, no vector index.
"""

import threading

import numpy as np
import psycopg
from pgvector import HalfVector

from app.db import connect
from app.eval.sets import Filters
from app.retrieval.embed import embedder
from app.schemas import Chunk

MODEL = "openai-large-768"  # the only embeddings loaded into Supabase
CHUNKS_NAME = "chunks-1000"
TOP_K = 20
FIELDS = tuple(Chunk.model_fields)  # every Chunk field is a column of chunks
FILTER_FIELDS = ("ticker", "fiscal_year", "fiscal_period", "form_type")


class SupabaseIndex:
    """Same interface as app.retrieval.search.Index: embed_query, prep, rank,
    search. One connection, opened on first use and shared by the app's threads."""

    model = MODEL

    def __init__(self) -> None:
        self.embed_query = embedder(MODEL, query=True)
        self._conn: psycopg.Connection | None = None
        self._lock = threading.Lock()

    def prep(self, vectors: np.ndarray) -> np.ndarray:
        """Round to 16-bit, as the stored halfvec embeddings are."""
        return vectors.astype(np.float16).astype(np.float32)

    def _query(self, sql: str, params: list) -> list[tuple]:
        with self._lock:
            for attempt in (1, 2):
                if self._conn is None or self._conn.closed:
                    self._conn = connect()
                try:
                    return self._conn.execute(sql, params).fetchall()
                except psycopg.OperationalError:
                    self._conn = None  # dropped (idle timeout, pooler restart)
                    if attempt == 2:
                        raise
        raise AssertionError("unreachable")

    def rank(
        self, query_vec: np.ndarray, k: int = TOP_K, filters: Filters | None = None
    ) -> list[tuple[Chunk, float]]:
        """Top k chunks by cosine similarity (1 - pgvector's cosine distance)."""
        where, params = [], []
        for field, value in (filters.model_dump() if filters else {}).items():
            if field in FILTER_FIELDS and value is not None:
                where.append(f"{field} = %s")
                params.append(value)
        sql = (
            f"SELECT {', '.join(FIELDS)}, embedding <=> %s AS distance "
            "FROM public.chunks "
            + (f"WHERE {' AND '.join(where)} " if where else "")
            + "ORDER BY distance LIMIT %s"
        )
        rows = self._query(sql, [HalfVector(query_vec), *params, k])
        return [
            (Chunk(**dict(zip(FIELDS, row[:-1], strict=True))), 1 - float(row[-1]))
            for row in rows
        ]

    def search(
        self, query: str, k: int = TOP_K, filters: Filters | None = None
    ) -> list[tuple[Chunk, float]]:
        vectors, _ = self.embed_query([query])
        return self.rank(self.prep(vectors)[0], k, filters)

    def count(self) -> int:
        return self._query("SELECT count(*) FROM public.chunks", [])[0][0]
