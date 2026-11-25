"""Reader check: the read-only login can read the tables and write nothing.

- Needs READER_DATABASE_URL in .env. Every write attempt is rolled back.

Run with:
    uv run python -m app.ingest.check_reader
"""

import os
import time

import psycopg
from dotenv import load_dotenv

from app.config import ENV_FILE

READS = {
    "read companies": "SELECT count(*) FROM public.companies",
    "read chunks": "SELECT count(*) FROM public.chunks",
    "read metrics": "SELECT count(*) FROM public.metrics",
}
INSERT = (
    "INSERT INTO public.companies (ticker, cik, name, fiscal_year_end) "
    "VALUES ('ZZZZ', '0000000001', 'test', '01-01')"
)
# Each must fail: (name, SQL, switch the transaction to read-write first)
BLOCKED = [
    ("insert, read-only default", INSERT, False),
    ("insert, without read-only", INSERT, True),
    ("update, without read-only", "UPDATE public.metrics SET value = 0", True),
    ("delete, without read-only", "DELETE FROM public.chunks", True),
    (
        "create table, without read-only",
        "CREATE TABLE public.reader_test (x int)",
        True,
    ),
    ("read auth.users", "SELECT * FROM auth.users LIMIT 1", False),
    ("query over 10 s", "SELECT pg_sleep(11)", False),
]
READ_ONLY_ERROR = "read-only transaction"


def main() -> None:
    load_dotenv(ENV_FILE)
    url = os.environ.get("READER_DATABASE_URL")
    if not url:
        raise SystemExit("Add READER_DATABASE_URL to .env first (see the docstring).")
    failures = 0
    with psycopg.connect(url, connect_timeout=10, autocommit=True) as conn:
        (user,) = conn.execute("SELECT current_user").fetchone()
        print(f"connected as {user}")
        for name, sql in READS.items():
            (count,) = conn.execute(sql).fetchone()
            ok = count > 0
            failures += not ok
            print(f"  {'ok  ' if ok else 'FAIL'} {name}: {count} rows")
        for name, sql, read_write in BLOCKED:
            t0 = time.perf_counter()
            try:
                with conn.transaction(force_rollback=True):
                    if read_write:
                        conn.execute("SET TRANSACTION READ WRITE")
                    conn.execute(sql)
                failures += 1
                print(f"  FAIL {name}: allowed")
            except psycopg.Error as exc:
                reason = exc.diag.message_primary or type(exc).__name__
                seconds = time.perf_counter() - t0
                # Without read-only, only the permissions should stop the write
                if read_write and READ_ONLY_ERROR in reason:
                    failures += 1
                    print(f"  FAIL {name}: read-only wasn't switched off ({reason})")
                else:
                    print(f"  ok   {name}: blocked ({reason}, {seconds:.1f} s)")
    print("all checks passed" if not failures else f"{failures} checks failed")


if __name__ == "__main__":
    main()
