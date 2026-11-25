"""The Supabase search and metrics lookup, against a fake connection: the SQL they
build and how rows become Chunks and Metrics. No network calls."""

from datetime import date
from decimal import Decimal

import numpy as np
import psycopg
import pytest
from conftest import CHUNK, REVENUE

from app.eval.sets import Filters
from app.retrieval import supabase
from app.retrieval.supabase import FIELDS, SupabaseIndex
from app.steps import tools


class FakeConn:
    def __init__(self, rows, fail_first: bool = False):
        self.rows, self.fail_first = rows, fail_first
        self.calls: list[tuple[str, list]] = []
        self.closed = False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if self.fail_first:
            self.fail_first = False
            raise psycopg.OperationalError("server closed the connection")
        return self

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


def chunk_row(distance: float) -> tuple:
    return (*(getattr(CHUNK, f) for f in FIELDS), distance)


@pytest.fixture
def conn(monkeypatch):
    fake = FakeConn([chunk_row(0.25)])
    monkeypatch.setattr(supabase, "connect", lambda: fake)
    return fake


def test_rank_filters_orders_by_distance_and_returns_chunks(conn):
    index = SupabaseIndex()
    filters = Filters(ticker="AAPL", fiscal_year=2024, fiscal_period=None)
    hits = index.rank(np.ones(768, dtype=np.float32), 20, filters)

    sql, params = conn.calls[0]
    assert "WHERE ticker = %s AND fiscal_year = %s ORDER BY distance LIMIT %s" in sql
    assert "embedding <=> %s AS distance" in sql
    assert params[1:] == ["AAPL", 2024, 20]  # the query vector comes first
    assert [(c.chunk_id, s) for c, s in hits] == [(CHUNK.chunk_id, 0.75)]
    assert hits[0][0] == CHUNK


def test_rank_without_filters_searches_every_chunk(conn):
    SupabaseIndex().rank(np.ones(768, dtype=np.float32), 5)
    sql, params = conn.calls[0]
    assert "WHERE" not in sql and len(params) == 2 and params[-1] == 5


def test_a_dropped_connection_is_reopened_once(monkeypatch):
    opened = []

    def connect():
        fake = FakeConn([chunk_row(0.1)], fail_first=not opened)
        opened.append(fake)
        return fake

    monkeypatch.setattr(supabase, "connect", connect)
    hits = SupabaseIndex().rank(np.ones(768, dtype=np.float32))
    assert len(opened) == 2 and len(hits) == 1


def test_prep_rounds_to_half_precision():
    v = np.array([[0.123456789]], dtype=np.float32)
    assert SupabaseIndex().prep(v)[0, 0] == np.float32(np.float16(0.123456789))


def test_metrics_reads_the_table_into_the_tools_lookup(monkeypatch):
    row = tuple(getattr(REVENUE, f) for f in tools.FIELDS)
    fake = FakeConn([row])
    monkeypatch.setattr("app.db.connect", lambda: fake)
    tools.metrics.cache_clear()
    try:
        got = tools.metrics()
    finally:
        tools.metrics.cache_clear()
    assert got == {("AAPL", "revenue", 2024, "FY"): REVENUE}
    assert got[("AAPL", "revenue", 2024, "FY")].value == Decimal("391035000000")
    assert got[("AAPL", "revenue", 2024, "FY")].period_end == date(2024, 9, 28)
    assert "FROM public.metrics" in fake.calls[0][0]
