// Client for the FastAPI backend. Types mirror app.pipeline.Response and the
// models it nests (backend/app/pipeline.py, tools.py, router.py, schemas.py).

export const API_URL: string =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// Same limits as backend/app/main.py
export const MAX_QUESTION_CHARS = 1000;
export const MAX_HISTORY = 20;

export type Route = "narrative" | "metric" | "hybrid" | "ambiguous" | "out_of_scope";
export type Behavior = "answer" | "clarify" | "refuse";
export type FiscalPeriod = "Q1" | "Q2" | "Q3" | "Q4" | "FY";
export type Unit = "USD" | "USD/shares";

export interface Citation {
  n: number; // the [n] in the answer
  kind: "filing" | "metric";
  ticker: string;
  accession_no: string;
  url: string; // the filing document, or the filing's folder on EDGAR for a metric
  section: string | null;
  chunk_id: string | null;
  metric: string | null;
  fiscal_year: number | null;
  fiscal_period: FiscalPeriod | null;
}

export interface MetricRow {
  ticker: string;
  fiscal_year: number;
  fiscal_period: FiscalPeriod;
  metric: string;
  value: string; // a Decimal, sent as a string
  unit: Unit;
  period_start: string | null;
  period_end: string;
  xbrl_tag: string;
  accession_no: string;
  filed: string;
  derived: boolean;
}

export interface MissingPeriod {
  ticker: string;
  fiscal_year: number;
  fiscal_period: FiscalPeriod;
}

export interface ToolResult {
  tool: string;
  args: Record<string, unknown>;
  rows: MetricRow[];
  missing: MissingPeriod[];
  chart: "table" | "bar" | "line";
  note: string | null;
}

export interface SearchFilters {
  ticker: string | null;
  fiscal_year: number | null;
  fiscal_period: FiscalPeriod | null;
}

/** A sentence with figures that its cited sources don't have (app.steps.citecheck). */
export interface Unsupported {
  sentence: string; // exactly as it appears in the answer
  numbers: string[]; // the unsupported figures, as written
  cited: number[]; // the sources it was checked against
}

export interface ChatResponse {
  run_id: string;
  question: string;
  standalone_question: string;
  route: Route;
  filters: SearchFilters;
  behavior: Behavior;
  answer: string; // with inline citations like [2]
  citations: Citation[];
  invalid_citations: number[];
  unsupported: Unsupported[];
  tool: ToolResult | null;
  ms: Record<string, number>;
}

export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

/** A failed request, with a message fit to show the user. */
export class ApiError extends Error {
  offline: boolean;

  constructor(message: string, offline = false) {
    super(message);
    this.name = "ApiError";
    this.offline = offline;
  }
}

async function errorMessage(res: Response): Promise<string> {
  const body = await res.json().catch(() => null);
  if (res.status === 400) {
    const details: { field: string; message: string }[] = body?.details ?? [];
    const text = details.map((d) => (d.field ? `${d.field}: ${d.message}` : d.message));
    return text.length ? `Bad request. ${text.join("; ")}` : "Bad request.";
  }
  if (res.status === 502) return "The model call failed. Try again.";
  if (res.status === 503) return "The database can't be reached. Try again in a minute.";
  return `The backend returned an error (${res.status}).`;
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(
      `Can't reach the backend at ${API_URL}. Is it running?`,
      true,
    );
  }
  if (!res.ok) throw new ApiError(await errorMessage(res));
  return res;
}

export async function chat(
  question: string,
  history: HistoryMessage[],
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const res = await request("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history: history.slice(-MAX_HISTORY) }),
    signal,
  });
  return res.json();
}

export async function health(signal?: AbortSignal): Promise<boolean> {
  try {
    const res = await request("/health", { signal });
    const body = await res.json();
    return body?.status === "ok";
  } catch {
    return false;
  }
}
