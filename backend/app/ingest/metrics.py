"""Metrics loader: CompanyFacts XBRL -> one row per company, metric and fiscal period.

- Fiscal periods come from each fact's dates, never XBRL's fy/fp tags.
- Missing quarters are derived from year-to-date values; Q4 = FY - Q1 - Q2 - Q3.
- For restated values, the latest filing wins.

Run with:
    uv run python -m app.ingest.metrics [--refresh]
"""

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from app.config import get_settings
from app.fiscal import fiscal_year, quarter
from app.ingest.extract import load_companies
from app.schemas import Company, Metric

ROOT = Path(__file__).resolve().parents[3]  # repo root
FACTS_DIR = ROOT / "data/raw/companyfacts"  # CIK##########.json from the SEC
OUT_DIR = ROOT / "data/processed/metrics"  # one <TICKER>.jsonl per company

# Each metric: (unit, kind, XBRL tags).
METRIC_TAGS = {
    "revenue": (
        "USD",
        "flow",
        [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
        ],
    ),
    "net_income": ("USD", "flow", ["NetIncomeLoss"]),
    "operating_income": ("USD", "flow", ["OperatingIncomeLoss"]),
    "operating_cash_flow": (
        "USD",
        "flow",
        [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ],
    ),
    "rnd_expense": (
        "USD",
        "flow",
        [
            "ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
        ],
    ),
    "total_assets": ("USD", "instant", ["Assets"]),
    "eps_diluted": (
        "USD/shares",
        "per_share",
        ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    ),
}
YEARS = range(2016, 2027)  # fiscal years kept
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}

# Classify a fact by its length in days: a quarter, 6- or 9-month year to date, or a
# full year. Ranges, not exact lengths, because 52/53-week years vary by a few days.
DURATIONS = [(80, 100, "3m"), (170, 200, "6m"), (260, 290, "9m"), (350, 380, "12m")]
# Year-to-date and full-year periods must end at this fiscal quarter. Anything else
# is skipped, e.g. AMZN's trailing-twelve-month totals that end mid-year.
ENDS_AT_QUARTER = {"6m": 2, "9m": 3, "12m": 4}


@dataclass
class Fact:
    """One reported XBRL value for a canonical metric, with its fiscal labels."""

    ticker: str
    metric: str
    tag: str
    unit: str
    length: str  # "instant", "3m", "6m" (YTD), "9m" (YTD), or "12m"
    fiscal_year: int
    fiscal_period: str  # "Q1"-"Q4" or "FY"; for YTD facts, the quarter they end
    period_start: date | None
    period_end: date
    value: Decimal
    accession_no: str
    filed: date


def period_length(start: date | None, end: date) -> str | None:
    """The fact's length: "instant", "3m", "6m", "9m", "12m", or None if none fit."""
    if start is None:
        return "instant"
    days = (end - start).days
    return next((name for lo, hi, name in DURATIONS if lo <= days <= hi), None)


def label(length: str, end: date, fye_month: int) -> str:
    """FY for a full year or year-end balance, else the quarter the fact ends in."""
    q = quarter(end, fye_month)
    if length == "12m" or (length == "instant" and q == 4):
        return "FY"
    return f"Q{q}"


def read_facts(company: Company) -> list[Fact]:
    """Every usable fact for our metrics from the company's saved CompanyFacts file."""
    path = FACTS_DIR / f"CIK{company.cik}.json"
    gaap = json.loads(path.read_text())["facts"]["us-gaap"]
    fye_month = int(company.fiscal_year_end[:2])
    facts = []
    for metric, (unit, _kind, tags) in METRIC_TAGS.items():
        for tag in tags:
            for f in gaap.get(tag, {}).get("units", {}).get(unit, []):
                if f.get("form") not in FORMS:
                    continue
                start = date.fromisoformat(f["start"]) if "start" in f else None
                end = date.fromisoformat(f["end"])
                length = period_length(start, end)
                if length is None:
                    continue
                if length in ENDS_AT_QUARTER and (
                    quarter(end, fye_month) != ENDS_AT_QUARTER[length]
                ):
                    continue
                facts.append(
                    Fact(
                        ticker=company.ticker,
                        metric=metric,
                        tag=tag,
                        unit=unit,
                        length=length,
                        fiscal_year=fiscal_year(end, fye_month),
                        fiscal_period=label(length, end, fye_month),
                        period_start=start,
                        period_end=end,
                        value=Decimal(str(f["val"])),
                        accession_no=f["accn"],
                        filed=date.fromisoformat(f["filed"]),
                    )
                )
    return facts


def pick_latest(facts: list[Fact]) -> list[Fact]:
    """Keep one fact per metric, period and length: the latest filed, so restated
    values win. If one filing has several tags, the first in METRIC_TAGS wins."""
    best: dict[tuple, Fact] = {}
    for f in facts:
        key = (f.metric, f.fiscal_year, f.fiscal_period, f.length)
        tags = METRIC_TAGS[f.metric][2]
        rank = (f.filed, -tags.index(f.tag))
        current = best.get(key)
        if current is None or rank > (current.filed, -tags.index(current.tag)):
            best[key] = f
    return list(best.values())


def to_metric(f: Fact, period: str, **changes) -> Metric:
    """Turn a fact into a Metric row; `changes` override fields for derived rows."""
    fields = {
        "ticker": f.ticker,
        "fiscal_year": f.fiscal_year,
        "fiscal_period": period,
        "metric": f.metric,
        "value": f.value,
        "unit": f.unit,
        "period_start": f.period_start,
        "period_end": f.period_end,
        "xbrl_tag": f.tag,
        "accession_no": f.accession_no,
        "filed": f.filed,
    }
    return Metric(**(fields | changes))


def subtract(total: Fact, parts: list[Metric], period: str) -> Metric:
    """A derived quarter: a year-to-date or full-year total minus earlier quarters."""
    return to_metric(
        total,
        period,
        value=total.value - sum(p.value for p in parts),
        period_start=parts[-1].period_end + timedelta(days=1),
        derived=True,
    )


def build_metrics(picked: list[Fact]) -> list[Metric]:
    """Build Metric rows for fiscal years 2016-2026.

    - Flows: reported quarters, or Q2/Q3 from year-to-date; Q4 = FY - Q1 - Q2 - Q3.
    - Total assets: Q4 is the year-end value.
    - EPS: reported quarters and FY only (no Q4).
    """
    by_key = {(f.metric, f.fiscal_year, f.fiscal_period, f.length): f for f in picked}
    rows = []
    for metric, (_unit, kind, _tags) in METRIC_TAGS.items():
        for year in YEARS:

            def get(period, length, m=metric, y=year):
                return by_key.get((m, y, period, length))

            if kind == "instant":
                quarters = [get(f"Q{q}", "instant") for q in (1, 2, 3)]
                rows += [to_metric(f, f"Q{q}") for q, f in enumerate(quarters, 1) if f]
                if fy := get("FY", "instant"):
                    rows.append(to_metric(fy, "FY"))
                    rows.append(to_metric(fy, "Q4", derived=True))
                continue

            q = {}  # quarter number -> Metric
            if f := get("Q1", "3m"):
                q[1] = to_metric(f, "Q1")
            if f := get("Q2", "3m"):
                q[2] = to_metric(f, "Q2")
            elif kind == "flow" and 1 in q and (ytd := get("Q2", "6m")):
                q[2] = subtract(ytd, [q[1]], "Q2")
            if f := get("Q3", "3m"):
                q[3] = to_metric(f, "Q3")
            elif kind == "flow" and 2 in q and (ytd := get("Q3", "9m")):
                six = get("Q2", "6m")
                parts = [to_metric(six, "Q2")] if six else [q[1], q[2]]
                q[3] = subtract(ytd, parts, "Q3")
            rows += q.values()
            if fy := get("FY", "12m"):
                rows.append(to_metric(fy, "FY"))
                if kind == "flow" and len(q) == 3:
                    rows.append(subtract(fy, [q[1], q[2], q[3]], "Q4"))
    return rows


def download_facts(company: Company) -> None:
    """Download the company's CompanyFacts file from the SEC (with --refresh)."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{company.cik}.json"
    headers = {"User-Agent": get_settings().sec_user_agent}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r:
        (FACTS_DIR / f"CIK{company.cik}.json").write_bytes(r.read())
    time.sleep(0.2)  # stay under SEC's 10 requests/second


def coverage_gaps(rows: list[Metric]) -> list[str]:
    """Periods a metric is missing that the company has for other metrics that year.
    A year with no 10-K yet isn't flagged, and EPS never needs Q4."""
    periods: dict[int, set[str]] = {}
    have: dict[tuple[str, int], set[str]] = {}
    for r in rows:
        periods.setdefault(r.fiscal_year, set()).add(r.fiscal_period)
        have.setdefault((r.metric, r.fiscal_year), set()).add(r.fiscal_period)
    gaps = []
    for metric in METRIC_TAGS:
        for year, expected in sorted(periods.items()):
            if metric == "eps_diluted":
                expected = expected - {"Q4"}
            missing = sorted(expected - have.get((metric, year), set()))
            if missing:
                gaps.append(f"{metric} FY{year}: {', '.join(missing)}")
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--refresh", action="store_true", help="download CompanyFacts again first"
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'':5} " + " ".join(f"{m[:10]:>10}" for m in METRIC_TAGS))
    all_gaps = []
    for company in load_companies():
        if args.refresh:
            download_facts(company)
        rows = build_metrics(pick_latest(read_facts(company)))
        out = OUT_DIR / f"{company.ticker}.jsonl"
        out.write_text("".join(r.model_dump_json() + "\n" for r in rows))
        counts = [sum(r.metric == m for r in rows) for m in METRIC_TAGS]
        print(f"{company.ticker:5} " + " ".join(f"{n:>10}" for n in counts))
        all_gaps += [f"{company.ticker} {g}" for g in coverage_gaps(rows)]
    print(f"\nRows per metric above; files in {OUT_DIR.relative_to(ROOT)}/")
    print(f"Gaps ({len(all_gaps)}):")
    for gap in all_gaps:
        print(f"  {gap}")


if __name__ == "__main__":
    main()
