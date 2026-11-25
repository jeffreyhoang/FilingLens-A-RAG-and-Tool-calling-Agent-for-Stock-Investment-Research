"""Data models for companies, text chunks, and financial metrics."""

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

CIK_PATTERN = r"^\d{10}$"  # 10 zero-padded digits

FormType = Literal["10-K", "10-Q"]
ChunkPeriod = Literal["Q1", "Q2", "Q3", "FY"]  # filings never cover Q4 alone
MetricPeriod = Literal["Q1", "Q2", "Q3", "Q4", "FY"]  # Q4 is always derived
MetricName = Literal[
    "revenue",
    "net_income",
    "operating_income",
    "operating_cash_flow",
    "rnd_expense",
    "total_assets",
    "eps_diluted",
]
Unit = Literal["USD", "USD/shares"]


class Company(BaseModel):
    ticker: str
    cik: str = Field(pattern=CIK_PATTERN)
    name: str
    fiscal_year_end: str = Field(pattern=r"^\d{2}-\d{2}$")  # MM-DD
    notes: str = ""


class Section(BaseModel):
    """One cleaned narrative section of a filing (extractor output)."""

    ticker: str
    cik: str = Field(pattern=CIK_PATTERN)
    form_type: FormType
    fiscal_year: int = Field(ge=2016, le=2026)
    fiscal_period: ChunkPeriod
    period_end: date
    filing_date: date
    accession_no: str
    section: str  # e.g. "Item 1A", "Part I, Item 2"
    section_title: str
    text: str
    source_url: str


class Chunk(Section):
    """A piece of a section's text (chunker output); `text` holds only this piece."""

    chunk_id: str  # "{accession_no}:{section}:{position}"
    position: int = Field(ge=0)
    # Where `text` sits in the section's text: section.text[char_start:char_end]
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)


class Metric(BaseModel):
    ticker: str
    fiscal_year: int = Field(ge=2016, le=2026)
    fiscal_period: MetricPeriod
    metric: MetricName
    value: Decimal
    unit: Unit
    period_start: date | None = None  # None for balance-sheet (instant) values
    period_end: date
    xbrl_tag: str
    accession_no: str
    filed: date
    derived: bool = False
