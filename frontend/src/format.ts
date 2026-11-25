import type { Citation, FiscalPeriod, Route, SearchFilters, Unit } from "./api.ts";

const METRIC_NAMES: Record<string, string> = {
  revenue: "Revenue",
  net_income: "Net income",
  operating_income: "Operating income",
  operating_cash_flow: "Operating cash flow",
  rnd_expense: "R&D expense",
  total_assets: "Total assets",
  eps_diluted: "Diluted EPS",
};

export function metricLabel(metric: string | null): string {
  if (!metric) return "Metric";
  return METRIC_NAMES[metric] ?? metric.replaceAll("_", " ");
}

/** "FY2024", or "Q3 FY2022" for a quarter. */
export function periodLabel(year: number | null, period: FiscalPeriod | null): string {
  if (year == null) return period ?? "";
  if (!period || period === "FY") return `FY${year}`;
  return `${period} FY${year}`;
}

const SIG3 = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 3 });
const PER_SHARE = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const FULL_USD = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const SCALES: [number, string][] = [
  [1e12, "T"],
  [1e9, "B"],
  [1e6, "M"],
  [1e3, "K"],
];

/** Dollars in millions or billions to 3 significant digits ("$391B"); EPS in dollars. */
export function formatValue(value: number, unit: Unit): string {
  if (!Number.isFinite(value)) return "—";
  if (unit === "USD/shares") return PER_SHARE.format(value);
  const sign = value < 0 ? "-" : "";
  const abs = Math.abs(value);
  for (const [size, suffix] of SCALES) {
    if (abs >= size) return `${sign}$${SIG3.format(abs / size)}${suffix}`;
  }
  return `${sign}$${SIG3.format(abs)}`;
}

/** The exact value, for tooltips ("$391,035,000,000"). */
export function formatExact(value: number, unit: Unit): string {
  if (!Number.isFinite(value)) return "—";
  return unit === "USD/shares" ? PER_SHARE.format(value) : FULL_USD.format(value);
}

const ROUTE_NAMES: Record<Route, string> = {
  narrative: "Filing text",
  metric: "Financial data",
  hybrid: "Filing text and financial data",
  ambiguous: "Needs more detail",
  out_of_scope: "Out of scope",
};

export function routeLabel(route: Route): string {
  return ROUTE_NAMES[route];
}

/** "AAPL · FY2024", or "" when the router set no filters. */
export function filtersLabel(filters: SearchFilters): string {
  const parts = [
    filters.ticker,
    filters.fiscal_year != null || filters.fiscal_period
      ? periodLabel(filters.fiscal_year, filters.fiscal_period)
      : null,
  ];
  return parts.filter(Boolean).join(" · ");
}

/** What a citation points to, in a few words. */
export function citationLabel(c: Citation): string {
  if (c.kind === "metric") {
    return `${c.ticker} ${metricLabel(c.metric)}, ${periodLabel(c.fiscal_year, c.fiscal_period)}`;
  }
  return c.section ? `${c.ticker} filing, ${c.section}` : `${c.ticker} filing`;
}
