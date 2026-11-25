"""Spot-check: compare metric rows with the XBRL in their source filings.

Run with:
    uv run python -m app.ingest.spot_check
"""

import random
from decimal import Decimal
from pathlib import Path

from app.config import get_settings
from app.ingest.extract import edgar, list_filings, load_companies
from app.schemas import Metric

ROOT = Path(__file__).resolve().parents[3]  # repo root
METRICS_DIR = ROOT / "data/processed/metrics"
SAMPLE_PER_COMPANY = 2
# Derived Q4 values to recompute: (ticker, metric, fiscal year)
Q4_CHECKS = [
    ("AAPL", "revenue", 2026),
    ("MSFT", "net_income", 2026),
    ("NVDA", "operating_cash_flow", 2026),
    ("CSCO", "operating_income", 2025),
]


def load_rows(ticker: str) -> list[Metric]:
    path = METRICS_DIR / f"{ticker}.jsonl"
    return [Metric.model_validate_json(line) for line in path.open()]


def filing_value(accession_no: str, tag: str, start, end) -> Decimal | None:
    """The undimensioned value of `tag` for the period in the filing's own XBRL."""
    df = edgar.find(accession_no).xbrl().facts.to_dataframe()
    df = df[(df["concept"] == f"us-gaap:{tag}") & df["dimension"].isna()]
    if start is None:
        df = df[df["period_instant"] == str(end)]
    else:
        df = df[(df["period_start"] == str(start)) & (df["period_end"] == str(end))]
    values = {Decimal(str(v)) for v in df["value"]}
    return values.pop() if len(values) == 1 else None


def check_reported(rng: random.Random) -> int:
    """Sample non-derived rows and confirm each value is in its source filing."""
    failures = 0
    for company in load_companies():
        rows = [r for r in load_rows(company.ticker) if not r.derived]
        for r in rng.sample(rows, SAMPLE_PER_COMPANY):
            found = filing_value(
                r.accession_no, r.xbrl_tag, r.period_start, r.period_end
            )
            ok = found == r.value
            failures += not ok
            status = "OK  " if ok else "FAIL"
            print(
                f"{status} {r.ticker:5} {r.metric:19} FY{r.fiscal_year} "
                f"{r.fiscal_period}: stored {r.value} | "
                f"filing {r.accession_no} has {found}"
            )
    return failures


def check_q4() -> int:
    """Recompute derived Q4 as the 10-K's full year minus the Q3 10-Q's 9 months."""
    companies = {c.ticker: c for c in load_companies()}
    failures = 0
    for ticker, metric, year in Q4_CHECKS:
        rows = load_rows(ticker)
        q4 = next(
            r
            for r in rows
            if (r.metric, r.fiscal_year, r.fiscal_period) == (metric, year, "Q4")
        )
        fy = next(
            r
            for r in rows
            if (r.metric, r.fiscal_year, r.fiscal_period) == (metric, year, "FY")
        )
        ref = next(
            f
            for f in list_filings(companies[ticker])
            if (f.fiscal_year, f.fiscal_period) == (year, "Q3")
        )
        full = filing_value(
            fy.accession_no, fy.xbrl_tag, fy.period_start, fy.period_end
        )
        nine = filing_value(
            ref.accession_no, fy.xbrl_tag, fy.period_start, ref.period_end
        )
        expected = full - nine if full is not None and nine is not None else None
        ok = expected == q4.value
        failures += not ok
        print(
            f"{'OK  ' if ok else 'FAIL'} {ticker:5} {metric:19} FY{year} Q4: stored "
            f"{q4.value} | 10-K {full} - Q3 10-Q 9 months {nine} = {expected}"
        )
    return failures


def main() -> None:
    edgar.set_identity(get_settings().sec_user_agent)
    reported = check_reported(random.Random(0))
    derived = check_q4()
    print(f"\nReported rows failing: {reported} | Derived Q4 failing: {derived}")


if __name__ == "__main__":
    main()
