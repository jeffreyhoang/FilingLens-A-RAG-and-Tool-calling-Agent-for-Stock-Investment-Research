"""Supabase loader: Loads companies, chunks, embeddings, metrics into Supabase


- Needs the owner's DATABASE_URL; the read-only login can't write.
- Each company is replaced in one transaction, so a rerun never half-loads it.

Run with:
    uv run python -m app.ingest.load_supabase [TICKER ...] [--check]
"""

import argparse
import json
import time
from collections.abc import Iterator

import numpy as np
import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector

from app.config import get_settings
from app.retrieval.embed import EMB_DIR, ROOT
from app.schemas import Chunk, Company, Metric

EMBED_MODEL = "openai-large-768"  # the setup chosen in T7 (and app.pipeline)
CHUNKS_DIR = ROOT / "data/processed/chunks-1000"
METRICS_DIR = ROOT / "data/processed/metrics"
COMPANIES = ROOT / "data/companies.json"
DIMS = 768

# Binary COPY types. cik is char(10), sent as text: psycopg has no binary dumper
# for char(n), and Postgres reads text bytes into it the same way.
CHUNK_COLUMNS = (
    ("chunk_id", "text"),
    ("ticker", "text"),
    ("cik", "text"),
    ("form_type", "text"),
    ("fiscal_year", "int4"),
    ("fiscal_period", "text"),
    ("period_end", "date"),
    ("filing_date", "date"),
    ("accession_no", "text"),
    ("section", "text"),
    ("section_title", "text"),
    ("position", "int4"),
    ("char_start", "int4"),
    ("char_end", "int4"),
    ("text", "text"),
    ("source_url", "text"),
    ("embedding", "halfvec"),
)
METRIC_COLUMNS = (
    ("ticker", "text"),
    ("fiscal_year", "int4"),
    ("fiscal_period", "text"),
    ("metric", "text"),
    ("value", "numeric"),
    ("unit", "text"),
    ("period_start", "date"),
    ("period_end", "date"),
    ("xbrl_tag", "text"),
    ("accession_no", "text"),
    ("filed", "date"),
    ("derived", "bool"),
)


def companies() -> list[Company]:
    return [Company.model_validate(c) for c in json.loads(COMPANIES.read_text())]


def chunks_with_vectors(ticker: str) -> tuple[list[Chunk], np.ndarray]:
    """A company's chunks and their embeddings, in the same order."""
    emb_dir = EMB_DIR / EMBED_MODEL / CHUNKS_DIR.name
    path = CHUNKS_DIR / f"{ticker}.jsonl"
    chunks = [Chunk.model_validate_json(line) for line in path.open() if line.strip()]
    ids = (emb_dir / f"{ticker}.ids.txt").read_text().splitlines()
    vectors = np.load(emb_dir / f"{ticker}.npy")
    if [c.chunk_id for c in chunks] != ids:
        raise ValueError(f"{ticker}: chunks changed since embedding")
    if vectors.shape != (len(chunks), DIMS):
        raise ValueError(f"{ticker}: {vectors.shape} vectors for {len(chunks)} chunks")
    return chunks, vectors


def metrics(ticker: str) -> list[Metric]:
    path = METRICS_DIR / f"{ticker}.jsonl"
    return [Metric.model_validate_json(line) for line in path.open() if line.strip()]


def chunk_rows(chunks: list[Chunk], vectors: np.ndarray) -> Iterator[tuple]:
    for chunk, vector in zip(chunks, vectors, strict=True):
        values = [getattr(chunk, name) for name, _ in CHUNK_COLUMNS[:-1]]
        yield (*values, HalfVector(vector))


def metric_rows(rows: list[Metric]) -> Iterator[tuple]:
    for m in rows:
        yield tuple(getattr(m, name) for name, _ in METRIC_COLUMNS)


def copy(cur: psycopg.Cursor, table: str, columns: tuple, rows: Iterator) -> None:
    names = ", ".join(name for name, _ in columns)
    sql = f"COPY public.{table} ({names}) FROM STDIN WITH (FORMAT BINARY)"
    with cur.copy(sql) as cp:
        cp.set_types([kind for _, kind in columns])
        for row in rows:
            cp.write_row(row)


def check(tickers: list[str]) -> None:
    """Validate the local files and count what would be loaded."""
    total_chunks = total_metrics = 0
    for ticker in tickers:
        chunks, _ = chunks_with_vectors(ticker)
        n_metrics = len(metrics(ticker))
        total_chunks += len(chunks)
        total_metrics += n_metrics
        print(f"  {ticker:6} {len(chunks):6} chunks  {n_metrics:5} metric rows")
    print(
        f"{len(tickers)} companies, {total_chunks} chunks, {total_metrics} metric rows"
    )


def load(tickers: list[str]) -> None:
    by_ticker = {c.ticker: c for c in companies()}
    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        conn.execute("SET search_path = public, extensions")
        conn.execute("SET statement_timeout = '5min'")
        register_vector(conn)
        with conn.transaction():
            conn.cursor().executemany(
                """INSERT INTO public.companies
                    (ticker, cik, name, fiscal_year_end, notes)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (ticker) DO UPDATE SET cik = excluded.cik,
                    name = excluded.name, fiscal_year_end = excluded.fiscal_year_end,
                    notes = excluded.notes""",
                [
                    (c.ticker, c.cik, c.name, c.fiscal_year_end, c.notes)
                    for c in by_ticker.values()
                ],
            )
        for ticker in tickers:
            t0 = time.perf_counter()
            chunks, vectors = chunks_with_vectors(ticker)
            rows = metrics(ticker)
            with conn.transaction(), conn.cursor() as cur:
                cur.execute("DELETE FROM public.chunks WHERE ticker = %s", (ticker,))
                cur.execute("DELETE FROM public.metrics WHERE ticker = %s", (ticker,))
                copy(cur, "chunks", CHUNK_COLUMNS, chunk_rows(chunks, vectors))
                copy(cur, "metrics", METRIC_COLUMNS, metric_rows(rows))
            seconds = time.perf_counter() - t0
            print(
                f"  {ticker:6} {len(chunks):6} chunks  {len(rows):5} metric rows  "
                f"{seconds:5.1f} s"
            )
        summary(conn)


def summary(conn: psycopg.Connection) -> None:
    for table in ("companies", "chunks", "metrics"):
        count, size = conn.execute(
            f"SELECT count(*), pg_size_pretty(pg_total_relation_size('public.{table}'))"
            f" FROM public.{table}"
        ).fetchone()
        print(f"{table:10} {count:7} rows  {size}")
    (db_size,) = conn.execute(
        "SELECT pg_size_pretty(pg_database_size(current_database()))"
    ).fetchone()
    print(f"database total {db_size} (free tier: 500 MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="*", help="default: all 10 companies")
    parser.add_argument("--check", action="store_true", help="files only, no database")
    args = parser.parse_args()
    known = [c.ticker for c in companies()]
    tickers = [t.upper() for t in args.tickers] or known
    unknown = sorted(set(tickers) - set(known))
    if unknown:
        parser.error(f"unknown tickers: {', '.join(unknown)}")
    if args.check:
        check(tickers)
    else:
        load(tickers)


if __name__ == "__main__":
    main()
