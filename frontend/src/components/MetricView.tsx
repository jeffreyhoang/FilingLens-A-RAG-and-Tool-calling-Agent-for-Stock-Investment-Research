import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MetricRow, ToolResult, Unit } from "../api.ts";
import { formatExact, formatValue, metricLabel, periodLabel } from "../format.ts";

const ACCENT = "#2563eb"; // blue-600
const GRID = "#e2e8f0"; // slate-200
const AXIS = "#64748b"; // slate-500

interface Point {
  label: string;
  value: number;
}

function Chart({ kind, data, unit, name }: { kind: "bar" | "line"; data: Point[]; unit: Unit; name: string }) {
  const axisProps = { tick: { fontSize: 12, fill: AXIS }, stroke: GRID };
  const yAxis = (
    <YAxis
      {...axisProps}
      width={64}
      tickFormatter={(v: number) => formatValue(v, unit)}
    />
  );
  const tooltip = (
    <Tooltip
      formatter={(v) => [formatValue(Number(v), unit), name]}
      contentStyle={{ fontSize: 12, borderRadius: 6, borderColor: GRID }}
    />
  );
  return (
    <div className="h-60 w-full" role="img" aria-label={`${name} chart`}>
      <ResponsiveContainer width="100%" height="100%">
        {kind === "bar" ? (
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="label" {...axisProps} />
            {yAxis}
            {tooltip}
            <Bar dataKey="value" name={name} fill={ACCENT} radius={[3, 3, 0, 0]} maxBarSize={56} />
          </BarChart>
        ) : (
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="label" {...axisProps} />
            {yAxis}
            {tooltip}
            <Line
              type="monotone"
              dataKey="value"
              name={name}
              stroke={ACCENT}
              strokeWidth={2}
              dot={{ r: 3, fill: ACCENT }}
            />
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

function ValuesTable({ rows }: { rows: MetricRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-xs text-slate-500">
            <th className="py-1.5 pr-3 font-medium">Company</th>
            <th className="py-1.5 pr-3 font-medium">Period</th>
            <th className="py-1.5 pr-3 font-medium">Period end</th>
            <th className="py-1.5 text-right font-medium">Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const value = Number(r.value);
            return (
              <tr key={`${r.ticker}-${r.fiscal_year}-${r.fiscal_period}`} className="border-b border-slate-100 last:border-0">
                <td className="py-1.5 pr-3 font-medium">{r.ticker}</td>
                <td className="py-1.5 pr-3">{periodLabel(r.fiscal_year, r.fiscal_period)}</td>
                <td className="py-1.5 pr-3 text-slate-500">{r.period_end}</td>
                <td className="py-1.5 text-right tabular-nums" title={formatExact(value, r.unit)}>
                  {formatValue(value, r.unit)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** The tool's rows as a table, bar chart (companies) or line chart (a series). */
export function MetricView({ tool }: { tool: ToolResult }) {
  const { rows, missing, note } = tool;
  if (!rows.length && !missing.length && !note) return null;

  const metric = rows[0]?.metric ?? (tool.args.metric as string | undefined) ?? null;
  const name = metricLabel(metric);
  const unit = rows[0]?.unit ?? "USD";
  const tickers = new Set(rows.map((r) => r.ticker));

  // A line chart needs one company; a bar chart needs one period. Otherwise a table.
  let kind: "table" | "bar" | "line" = tool.chart;
  if (rows.length < 2) kind = "table";
  if (kind === "line" && tickers.size > 1) kind = "table";

  const data: Point[] = rows.map((r) => ({
    label: kind === "bar" ? r.ticker : periodLabel(r.fiscal_year, r.fiscal_period),
    value: Number(r.value),
  }));

  let subtitle = "";
  if (rows.length) {
    if (kind === "line") subtitle = [...tickers][0];
    else if (kind === "bar") subtitle = periodLabel(rows[0].fiscal_year, rows[0].fiscal_period);
  }

  return (
    <section className="mt-4 rounded-lg border border-slate-200 bg-slate-50/60 p-3">
      <h3 className="mb-2 text-sm font-semibold text-slate-800">
        {name}
        {subtitle && <span className="font-normal text-slate-500"> · {subtitle}</span>}
      </h3>
      {kind !== "table" && <Chart kind={kind} data={data} unit={unit} name={name} />}
      {rows.length > 0 && <ValuesTable rows={rows} />}
      {note && <p className="mt-2 text-xs text-slate-600">{capitalize(note)}.</p>}
      {missing.length > 0 && (
        <p className="mt-2 text-xs text-amber-800">
          No data for{" "}
          {missing.map((m) => `${m.ticker} ${periodLabel(m.fiscal_year, m.fiscal_period)}`).join(", ")}.
        </p>
      )}
    </section>
  );
}
