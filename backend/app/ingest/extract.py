"""Extractor: 10-K and 10-Q filings -> cleaned narrative sections.

Run with:
    uv run python -m app.ingest.extract
"""

import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import edgar

from app.config import get_settings
from app.fiscal import fiscal_period, fiscal_year
from app.schemas import Company, Section

ROOT = Path(__file__).resolve().parents[3]  # repo root
OUT_DIR = ROOT / "data/processed/sections"  # one <TICKER>.jsonl per company
os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", str(ROOT / "data/raw/edgartools"))

FORMS = ["10-K", "10-Q"]  # amendments are skipped
YEARS = range(2016, 2027)  # fiscal years 2016-2026

MDA = "Management's Discussion and Analysis"
# Sections to extract per form, keyed by EdgarTools item key -> section_title
SECTIONS = {
    "10-K": {"Item 1": "Business", "Item 1A": "Risk Factors", "Item 7": MDA},
    "10-Q": {"Part I, Item 2": MDA, "Part II, Item 1A": "Risk Factors"},
}

SHORT_SECTION = 1_000

# Whole lines that are page furniture, not content
JUNK_LINES = [
    # page footer, e.g. "Apple Inc. | 2021 Form 10-K | 20"
    r".{1,80}\|\s*(\d{4}\s+)?Form 10-[KQ]\s*\|\s*\d+",
    r"\d{1,3}",  # bare page number
    r"table of contents",
    r"part i{1,3}",  # running page header "PART II"
    r"item \d+[a-c]?\.?",  # running page header "Item 1A" with no title
]
JUNK_RE = re.compile(r"^\s*(?:" + "|".join(JUNK_LINES) + r")\s*$", re.IGNORECASE)
TITLE = (
    r"(?:Business|Risk Factors|Management.s Discussion and Analysis of Financial"
    r"\s+Condition and Results of Operations)"
)
# The section's own header at the start of its text, possibly clipped ("TEM 7.") or
# wrapped over two lines: "Item 1A. Risk Factors", "ITEM 2. MANAGEMENT'S ... OPERATIONS"
HEADER_RE = re.compile(rf"^I?TEM\s+\d+[A-C]?\.?\s*{TITLE}?", re.IGNORECASE)
# A title left alone on the first line, when the "Item N." line was dropped as junk
TITLE_LINE_RE = re.compile(rf"^{TITLE}\.?[ \t]*\n", re.IGNORECASE)


@dataclass
class FilingRef:
    """One filing to extract, with its fiscal labels."""

    company: Company
    form_type: str
    accession_no: str
    filing_date: date
    period_end: date
    fiscal_year: int
    fiscal_period: str
    filing: edgar.Filing


def load_companies() -> list[Company]:
    data = json.loads((ROOT / "data/companies.json").read_text())
    return [Company(**c) for c in data]


def list_filings(company: Company) -> list[FilingRef]:
    """The company's 10-K and 10-Q filings in fiscal years 2016-2026, oldest first."""
    fye_month = int(company.fiscal_year_end[:2])
    filings = edgar.Company(company.cik).get_filings(form=FORMS, amendments=False)
    # Read the period from the filing list's reportDate column. Filing.period_of_report
    # would download each filing's full text just to get the date.
    report_dates = filings.data.column("reportDate").to_pylist()
    refs = []
    for i, report_date in enumerate(report_dates):
        period_end = date.fromisoformat(str(report_date))
        year = fiscal_year(period_end, fye_month)
        if year not in YEARS:
            continue
        f = filings[i]
        refs.append(
            FilingRef(
                company=company,
                form_type=f.form,
                accession_no=f.accession_no,
                filing_date=f.filing_date,
                period_end=period_end,
                fiscal_year=year,
                fiscal_period=fiscal_period(period_end, fye_month, f.form),
                filing=f,
            )
        )
    return sorted(refs, key=lambda r: r.period_end)


def clean(text: str) -> str:
    """Drop page furniture and the section's own header; normalize whitespace."""
    text = text.replace("\xa0", " ")
    lines = [line.strip() for line in text.splitlines() if not JUNK_RE.match(line)]
    text = "\n".join(lines).strip()
    text = HEADER_RE.sub("", text, count=1).lstrip()  # section_title carries it
    text = TITLE_LINE_RE.sub("", text, count=1)  # (S1-S3)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_sections(ref: FilingRef) -> list[Section]:
    """The filing's narrative sections, cleaned. Missing or empty ones are left out."""
    doc = ref.filing.obj()
    sections = []
    for key, title in SECTIONS[ref.form_type].items():
        text = clean(doc[key] or "")
        if not text:
            continue
        sections.append(
            Section(
                ticker=ref.company.ticker,
                cik=ref.company.cik,
                form_type=ref.form_type,
                fiscal_year=ref.fiscal_year,
                fiscal_period=ref.fiscal_period,
                period_end=ref.period_end,
                filing_date=ref.filing_date,
                accession_no=ref.accession_no,
                section=key,
                section_title=title,
                text=text,
                source_url=ref.filing.document.url,
            )
        )
    return sections


def validate(refs: list[FilingRef], sections: list[Section]) -> list[dict]:
    """Flag missing or short sections and periods with more than one filing."""
    lengths = {(s.accession_no, s.section): len(s.text) for s in sections}
    per_period = Counter((r.fiscal_year, r.fiscal_period) for r in refs)
    flags = []
    for r in refs:
        base = {
            "ticker": r.company.ticker,
            "form_type": r.form_type,
            "fiscal_year": r.fiscal_year,
            "fiscal_period": r.fiscal_period,
            "accession_no": r.accession_no,
        }
        if per_period[(r.fiscal_year, r.fiscal_period)] > 1:
            flags.append({**base, "section": "", "issue": "duplicate period"})
        for key in SECTIONS[r.form_type]:
            chars = lengths.get((r.accession_no, key), 0)
            if chars == 0:
                flags.append({**base, "section": key, "issue": "missing"})
            elif chars < SHORT_SECTION:
                flags.append({**base, "section": key, "issue": f"short ({chars})"})
    return flags


def write_report(flags: list[dict]) -> Path:
    """Write all flags to validation_report.csv in OUT_DIR."""
    path = OUT_DIR / "validation_report.csv"
    fields = ["ticker", "form_type", "fiscal_year", "fiscal_period", "accession_no"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*fields, "section", "issue"])
        writer.writeheader()
        writer.writerows(flags)
    return path


def main(tickers: list[str]) -> None:
    """Extract, validate, and write every company; limit to `tickers` if given."""
    edgar.set_identity(get_settings().sec_user_agent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    companies = [c for c in load_companies() if not tickers or c.ticker in tickers]
    all_flags = []
    for company in companies:
        refs = list_filings(company)
        sections = [s for r in refs for s in extract_sections(r)]
        out = OUT_DIR / f"{company.ticker}.jsonl"
        out.write_text("".join(s.model_dump_json() + "\n" for s in sections))
        flags = validate(refs, sections)
        all_flags += flags
        print(
            f"{company.ticker}: {len(refs)} filings, {len(sections)} sections, "
            f"{len(flags)} flags -> {out.relative_to(ROOT)}"
        )
    print(
        f"Report: {write_report(all_flags).relative_to(ROOT)} ({len(all_flags)} flags)"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
