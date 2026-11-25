"""Metric tools: look up reported financial metrics.

- get_metric: one company, one metric, one period
- compare_companies: two or more companies, one metric, one period
- get_metric_series: one company, one metric, a range of fiscal years
- fiscal_year=None means the most recent year with data.

Run with:
    uv run python -m app.steps.tools TOOL 'JSON_ARGS'
"""

import argparse
import json
from collections.abc import Callable
from functools import cache
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas import Metric, MetricName, MetricPeriod
from app.steps.router import TICKERS, Ticker

Key = tuple[str, str, int, str]  # (ticker, metric, fiscal_year, fiscal_period)
FIELDS = tuple(Metric.model_fields)  # every Metric field is a column of metrics


@cache
def metrics() -> dict[Key, Metric]:
    """Every metric row, keyed for lookup; read from Supabase once (3,622 rows,
    loaded by app.ingest.load_supabase from data/processed/metrics/)."""
    from app.db import connect  # only when the data is needed

    with connect() as conn:
        rows = conn.execute(
            f"SELECT {', '.join(FIELDS)} FROM public.metrics WHERE ticker = ANY(%s)",
            [list(TICKERS)],
        ).fetchall()
    out = {}
    for row in rows:
        m = Metric(**dict(zip(FIELDS, row, strict=True)))
        out[(m.ticker, m.metric, m.fiscal_year, m.fiscal_period)] = m
    return out


class GetMetricArgs(BaseModel):
    ticker: Ticker
    metric: MetricName
    fiscal_year: int | None = Field(description="None for the most recent year")
    fiscal_period: MetricPeriod


class CompareCompaniesArgs(BaseModel):
    tickers: list[Ticker] = Field(description="Two or more companies")
    metric: MetricName
    fiscal_year: int | None = Field(description="None for the most recent year")
    fiscal_period: MetricPeriod


class GetMetricSeriesArgs(BaseModel):
    ticker: Ticker
    metric: MetricName
    start_year: int
    end_year: int
    fiscal_period: MetricPeriod


class Missing(BaseModel):
    ticker: str
    fiscal_year: int | None  # None: no year has this metric and period
    fiscal_period: MetricPeriod


class ToolResult(BaseModel):
    tool: str
    args: dict
    rows: list[Metric]  # in the order asked: tickers, or years ascending
    missing: list[Missing]
    chart: Literal["table", "bar", "line"]
    note: str | None = None  # e.g. which year was used when none was given


def latest_year(tickers: list[str], metric: str, period: str) -> int | None:
    """The most recent fiscal year in which every ticker has this metric and period."""
    years = [
        {y for (t, m, y, p) in metrics() if t == ticker and m == metric and p == period}
        for ticker in tickers
    ]
    common = set.intersection(*years) if years else set()
    return max(common) if common else None


def lookup(
    tickers: list[str], metric: str, years: list[int], period: str
) -> tuple[list[Metric], list[Missing]]:
    rows, missing = [], []
    for ticker in tickers:
        for year in years:
            row = metrics().get((ticker, metric, year, period))
            if row:
                rows.append(row)
            else:
                missing.append(
                    Missing(ticker=ticker, fiscal_year=year, fiscal_period=period)
                )
    return rows, missing


def one_period(
    tool: str, args: BaseModel, tickers: list[str], chart: str
) -> ToolResult:
    """Shared by get_metric and compare_companies: one metric, one period."""
    year, note = args.fiscal_year, None
    if year is None:
        year = latest_year(tickers, args.metric, args.fiscal_period)
        if year is None:
            missing = [
                Missing(ticker=t, fiscal_year=None, fiscal_period=args.fiscal_period)
                for t in tickers
            ]
            return ToolResult(
                tool=tool, args=args.model_dump(), rows=[], missing=missing,
                chart=chart, note="no fiscal year has this metric for all companies",
            )  # fmt: skip
        note = f"no year given; used fiscal {year}, the most recent with data"
    rows, missing = lookup(tickers, args.metric, [year], args.fiscal_period)
    return ToolResult(
        tool=tool, args=args.model_dump(), rows=rows, missing=missing,
        chart=chart, note=note,
    )  # fmt: skip


def get_metric(args: GetMetricArgs) -> ToolResult:
    """One company's metric for one fiscal period."""
    return one_period("get_metric", args, [args.ticker], "table")


def compare_companies(args: CompareCompaniesArgs) -> ToolResult:
    """One metric for one fiscal period across two or more companies."""
    tickers = list(dict.fromkeys(args.tickers))  # drop repeats, keep order
    return one_period("compare_companies", args, tickers, "bar")


def get_metric_series(args: GetMetricSeriesArgs) -> ToolResult:
    """One company's metric for each fiscal year from start_year to end_year."""
    lo, hi = sorted((args.start_year, args.end_year))
    rows, missing = lookup(
        [args.ticker], args.metric, list(range(lo, hi + 1)), args.fiscal_period
    )
    return ToolResult(
        tool="get_metric_series", args=args.model_dump(), rows=rows,
        missing=missing, chart="line",
    )  # fmt: skip


TOOLS: dict[str, tuple[type[BaseModel], Callable[..., ToolResult]]] = {
    "get_metric": (GetMetricArgs, get_metric),
    "compare_companies": (CompareCompaniesArgs, compare_companies),
    "get_metric_series": (GetMetricSeriesArgs, get_metric_series),
}


def run_tool(name: str, args: dict) -> ToolResult:
    """Validate the arguments for a tool by name and run it."""
    model, fn = TOOLS[name]
    return fn(model.model_validate(args))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tool", choices=list(TOOLS))
    parser.add_argument("args", help="the arguments as JSON")
    args = parser.parse_args()
    print(run_tool(args.tool, json.loads(args.args)).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
