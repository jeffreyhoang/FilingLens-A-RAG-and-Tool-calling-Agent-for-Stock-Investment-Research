"""Sets: formats for the test sets and for a system's predictions.

- Each set is eval/sets/<name>.jsonl; predictions match its items by `id`.
"""

from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from app.schemas import ChunkPeriod, FormType, MetricName, MetricPeriod

ROOT = Path(__file__).resolve().parents[3]  # repo root
SETS_DIR = ROOT / "eval/sets"

Route = Literal["narrative", "metric", "hybrid", "ambiguous", "out_of_scope"]
Behavior = Literal["answer", "refuse", "clarify"]
ToolName = Literal["get_metric", "compare_companies", "get_metric_series"]


class Evidence(BaseModel):
    """A passage that answers a query: an exact quote from one filing section."""

    accession_no: str
    section: str  # e.g. "Item 1A", "Part I, Item 2"
    quote: str  # copied verbatim from the section text


class MetricRef(BaseModel):
    """One row of data/processed/metrics/, and its value."""

    ticker: str
    metric: MetricName
    fiscal_year: int
    fiscal_period: MetricPeriod
    value: Decimal


class Filters(BaseModel):
    """Metadata a question names explicitly; None means not stated."""

    ticker: str | None = None
    fiscal_year: int | None = None
    fiscal_period: ChunkPeriod | None = None
    form_type: FormType | None = None


class RetrievalItem(BaseModel):
    id: str
    query: str
    filters: Filters = Filters()
    evidence: list[Evidence]  # any chunk overlapping any quote is relevant


class RouterItem(BaseModel):
    id: str
    query: str
    route: Route


class ToolItem(BaseModel):
    id: str
    query: str
    tool: ToolName
    args: dict[str, Any]  # argument names are provisional until the tools exist
    expected: list[MetricRef]  # the values the tool call should return


class GenerationItem(BaseModel):
    id: str
    query: str
    behavior: Behavior
    reference_answer: str | None = None  # None for refuse or clarify
    evidence: list[Evidence] = []
    metrics: list[MetricRef] = []


SafetyCategory = Literal["advice", "injection", "coverage", "period", "fabrication"]


class SafetyItem(BaseModel):
    """A question that tests a safety boundary. Any of `behaviors` is right. For
    answer items, ticker, fiscal_year and fiscal_period are what the sources the
    answer was written from should match (None: not checked)."""

    id: str
    query: str
    category: SafetyCategory
    behaviors: list[Behavior]
    ticker: str | None = None
    fiscal_year: int | None = None
    fiscal_period: MetricPeriod | None = None
    note: str = ""  # what the item tests


SETS: dict[str, type[BaseModel]] = {
    "retrieval": RetrievalItem,
    "router": RouterItem,
    "tool": ToolItem,
    "generation": GenerationItem,
    "safety": SafetyItem,
}


class RetrievedChunk(BaseModel):
    """A chunk a system returned, located by its section and character range."""

    accession_no: str
    section: str
    char_start: int
    char_end: int


class Prediction(BaseModel):
    """A system's output for one item. Fill only the fields the set scores."""

    id: str
    retrieved: list[RetrievedChunk] | None = None  # best first (retrieval)
    route: Route | None = None  # router
    filters: Filters | None = None  # router, scored on the retrieval set's labels
    tool: str | None = None  # tool
    args: dict[str, Any] | None = None  # tool
    answer: str | None = None  # generation
    behavior: Behavior | None = None  # generation
    citations: list[dict[str, Any]] | None = None  # generation: cited sources
    invalid_citations: list[int] | None = None  # generation: numbers with no source
    unsupported: list[dict[str, Any]] | None = None  # app.steps.citecheck
    # safety: each source's kind, ticker, fiscal_year, fiscal_period
    sources: list[dict[str, Any]] | None = None


def load_set(name: str) -> list[BaseModel]:
    """Read and validate eval/sets/<name>.jsonl."""
    model = SETS[name]
    path = SETS_DIR / f"{name}.jsonl"
    return [model.model_validate_json(line) for line in path.open() if line.strip()]


def load_predictions(path: Path) -> dict[str, Prediction]:
    """Read a predictions JSONL file, keyed by item id."""
    rows = [
        Prediction.model_validate_json(line) for line in path.open() if line.strip()
    ]
    return {p.id: p for p in rows}
