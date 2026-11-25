"""Label a period with its fiscal year and quarter from its end date and the
company's fiscal-year-end month."""

import calendar
from datetime import date, timedelta

GRACE = timedelta(days=7)  # 52/53-week years can end a few days into the next month


def fiscal_year_end(year: int, month: int) -> date:
    """Last day of the fiscal-year-end month in a given calendar year."""
    return date(year, month, calendar.monthrange(year, month)[1])


def fiscal_year(end: date, fye_month: int) -> int:
    """The fiscal year a period ending on `end` belongs to, named by the calendar
    year it ends in (MSFT FY2026 runs July 2025 to June 2026).

    The first year whose fiscal year end (plus the grace days) is on or after `end`.
    Computed from dates because XBRL's own `fy` tag is sometimes wrong.
    """
    year = end.year - 1
    while end > fiscal_year_end(year, fye_month) + GRACE:
        year += 1
    return year


def quarter(end: date, fye_month: int) -> int:
    """Fiscal quarter (1-4) a period ending on `end` closes: days since the previous
    fiscal year end / 91.3, rounded. 4 means it ends at the fiscal year end."""
    previous_end = fiscal_year_end(fiscal_year(end, fye_month) - 1, fye_month)
    return round((end - previous_end).days / 91.3)


def fiscal_period(end: date, fye_month: int, form_type: str) -> str:
    """A filing's period: "FY" for a 10-K, "Q1"-"Q3" for a 10-Q."""
    if form_type == "10-K":
        return "FY"
    return f"Q{quarter(end, fye_month)}"
