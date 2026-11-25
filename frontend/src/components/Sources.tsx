import type { Citation } from "../api.ts";
import { metricLabel, periodLabel } from "../format.ts";

function Source({ c }: { c: Citation }) {
  const what =
    c.kind === "metric"
      ? `${metricLabel(c.metric)}, ${periodLabel(c.fiscal_year, c.fiscal_period)}`
      : (c.section ?? "Filing text");
  return (
    <li className="flex gap-2">
      <span className="mt-0.5 inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded bg-slate-100 px-1 text-xs font-semibold text-slate-600">
        {c.n}
      </span>
      <div className="min-w-0">
        <p className="text-sm text-slate-800">
          <span className="font-medium">{c.ticker}</span> · {what}
        </p>
        <p className="text-xs break-all text-slate-500">
          Accession {c.accession_no} ·{" "}
          <a
            href={c.url}
            target="_blank"
            rel="noreferrer"
            className="text-blue-700 underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-blue-600"
          >
            {c.kind === "metric" ? "Open filing on EDGAR" : "Open filing"}
            <span aria-hidden> ↗</span>
          </a>
        </p>
      </div>
    </li>
  );
}

/** The cited sources under an answer, plus any citation numbers that match none. */
export function Sources({
  citations,
  invalid,
}: {
  citations: Citation[];
  invalid: number[];
}) {
  const unique = [...new Map(citations.map((c) => [c.n, c])).values()].sort(
    (a, b) => a.n - b.n,
  );
  if (!unique.length && !invalid.length) return null;
  return (
    <section className="mt-4 border-t border-slate-100 pt-3">
      <h3 className="mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">
        Sources
      </h3>
      <ol className="space-y-2">
        {unique.map((c) => (
          <Source key={c.n} c={c} />
        ))}
      </ol>
      {invalid.length > 0 && (
        <p className="mt-2 text-xs text-slate-500">
          {invalid.length === 1 ? "Citation" : "Citations"}{" "}
          {invalid.map((n) => `[${n}]`).join(", ")}{" "}
          {invalid.length === 1 ? "matches" : "match"} no source.
        </p>
      )}
    </section>
  );
}
