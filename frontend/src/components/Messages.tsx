import type { ChatResponse } from "../api.ts";
import { filtersLabel, routeLabel } from "../format.ts";
import { AnswerText } from "./AnswerText.tsx";
import { MetricView } from "./MetricView.tsx";
import { Sources } from "./Sources.tsx";

export function UserMessage({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <p className="max-w-[85%] rounded-2xl rounded-br-sm bg-blue-600 px-4 py-2.5 whitespace-pre-wrap text-white">
        {text}
      </p>
    </div>
  );
}

function Meta({ response }: { response: ChatResponse }) {
  const filters = filtersLabel(response.filters);
  const rewritten =
    response.standalone_question.trim() !== response.question.trim()
      ? response.standalone_question
      : null;
  return (
    <div className="mb-2 space-y-0.5 text-xs text-slate-500">
      <p>
        {routeLabel(response.route)}
        {filters && ` · ${filters}`}
      </p>
      {rewritten && <p>Answered as: “{rewritten}”</p>}
    </div>
  );
}

function CheckWarning({ response }: { response: ChatResponse }) {
  const numbers = [...new Set(response.unsupported.flatMap((u) => u.numbers))];
  return (
    <p className="mt-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
      <span aria-hidden>⚠ </span>
      {numbers.join(", ")} {numbers.length === 1 ? "wasn't" : "weren't"} found in the
      cited sources (underlined above). It may be computed from them, or wrong; check
      the filing before relying on it.
    </p>
  );
}

export function AssistantMessage({ response }: { response: ChatResponse }) {
  if (response.behavior !== "answer") {
    const clarify = response.behavior === "clarify";
    return (
      <article
        className={`max-w-[95%] rounded-2xl rounded-bl-sm border px-4 py-3 ${
          clarify ? "border-amber-200 bg-amber-50" : "border-slate-200 bg-slate-100"
        }`}
      >
        <p
          className={`mb-1 text-xs font-semibold tracking-wide uppercase ${
            clarify ? "text-amber-800" : "text-slate-600"
          }`}
        >
          {clarify ? "Needs more detail" : "Can't answer this"}
        </p>
        <div className="text-slate-800">
          <AnswerText
            text={response.answer}
            citations={response.citations}
            unsupported={response.unsupported}
          />
        </div>
        {response.unsupported?.length > 0 && <CheckWarning response={response} />}
        {/* A decline can still cite what the filings do say */}
        <Sources citations={response.citations} invalid={response.invalid_citations} />
      </article>
    );
  }
  return (
    <article className="max-w-[95%] rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3 shadow-sm">
      <Meta response={response} />
      <AnswerText
        text={response.answer}
        citations={response.citations}
        unsupported={response.unsupported}
      />
      {response.unsupported?.length > 0 && <CheckWarning response={response} />}
      {response.tool && <MetricView tool={response.tool} />}
      <Sources citations={response.citations} invalid={response.invalid_citations} />
    </article>
  );
}

export function ErrorMessage({ text }: { text: string }) {
  return (
    <p
      role="alert"
      className="max-w-[95%] rounded-2xl rounded-bl-sm border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
    >
      {text}
    </p>
  );
}

export function Loading() {
  return (
    <div className="flex items-center gap-2 px-1 text-sm text-slate-500" role="status">
      <span className="flex gap-1" aria-hidden>
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400" />
      </span>
      Searching filings…
    </div>
  );
}
