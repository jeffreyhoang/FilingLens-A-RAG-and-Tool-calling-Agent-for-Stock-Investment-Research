import type { ReactNode } from "react";
import type { Citation, Unsupported } from "../api.ts";
import { citationLabel } from "../format.ts";

// [2], or several numbers in one bracket: [1, 3]
const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

function Marker({ n, citation }: { n: number; citation?: Citation }) {
  if (!citation) {
    // A cited number that matches no source: shown, never linked
    return (
      <span className="text-slate-400" title="This citation matches no source">
        [{n}]
      </span>
    );
  }
  return (
    <a
      href={citation.url}
      target="_blank"
      rel="noreferrer"
      title={`${citationLabel(citation)} (opens the filing on EDGAR)`}
      aria-label={`Source ${n}: ${citationLabel(citation)}`}
      className="mx-0.5 inline-flex h-5 min-w-5 -translate-y-0.5 items-center justify-center rounded bg-blue-50 px-1 align-baseline text-xs font-semibold text-blue-700 ring-1 ring-blue-200 hover:bg-blue-100 focus-visible:outline-2 focus-visible:outline-blue-600"
    >
      {n}
    </a>
  );
}

/** Text with each [n] turned into a link to its source. */
function withMarkers(text: string, byN: Map<number, Citation>, key: string): ReactNode[] {
  const parts: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(MARKER)) {
    const start = match.index ?? 0;
    if (start > last) parts.push(text.slice(last, start));
    const numbers = match[1].split(",").map((s) => Number(s.trim()));
    parts.push(
      <span key={`${key}-${start}`} className="whitespace-nowrap">
        {numbers.map((n, i) => (
          <Marker key={`${n}-${i}`} n={n} citation={byN.get(n)} />
        ))}
      </span>,
    );
    last = start + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

interface Segment {
  text: string;
  flag?: Unsupported;
}

/** Split the answer into plain text and the flagged sentences, in order. */
function segments(text: string, flags: Unsupported[]): Segment[] {
  const found = flags
    .map((flag) => ({ flag, start: text.indexOf(flag.sentence) }))
    .filter((f) => f.start >= 0)
    .sort((a, b) => a.start - b.start);
  const out: Segment[] = [];
  let last = 0;
  for (const { flag, start } of found) {
    if (start < last) continue; // overlaps an earlier flag
    if (start > last) out.push({ text: text.slice(last, start) });
    out.push({ text: flag.sentence, flag });
    last = start + flag.sentence.length;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}

/** The answer, with citation links and a warning on each flagged sentence. */
export function AnswerText({
  text,
  citations,
  unsupported = [],
}: {
  text: string;
  citations: Citation[];
  unsupported?: Unsupported[];
}) {
  const byN = new Map(citations.map((c) => [c.n, c]));
  return (
    <p className="leading-relaxed whitespace-pre-wrap">
      {segments(text, unsupported).map((seg, i) =>
        seg.flag ? (
          <span
            key={i}
            className="rounded-sm bg-amber-50 underline decoration-amber-500 decoration-wavy underline-offset-4"
            title={`Not found in the cited sources: ${seg.flag.numbers.join(", ")}`}
          >
            {withMarkers(seg.text, byN, `s${i}`)}
            <span className="sr-only">
              {" "}
              (warning: {seg.flag.numbers.join(", ")} not found in the cited sources)
            </span>
          </span>
        ) : (
          <span key={i}>{withMarkers(seg.text, byN, `s${i}`)}</span>
        ),
      )}
    </p>
  );
}
