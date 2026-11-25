import { useState, type FormEvent, type KeyboardEvent } from "react";
import { MAX_QUESTION_CHARS } from "../api.ts";

/** The question box: Enter sends, Shift+Enter adds a line. */
export function Composer({
  disabled,
  onSend,
}: {
  disabled: boolean;
  onSend: (question: string) => void;
}) {
  const [text, setText] = useState("");
  const question = text.trim();
  const canSend = !disabled && question.length > 0;

  function submit(event?: FormEvent) {
    event?.preventDefault();
    if (!canSend) return;
    onSend(question);
    setText("");
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  const remaining = MAX_QUESTION_CHARS - text.length;
  return (
    <form onSubmit={submit} className="flex items-end gap-2">
      <label htmlFor="question" className="sr-only">
        Ask about a company's filings
      </label>
      <div className="relative flex-1">
        <textarea
          id="question"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          maxLength={MAX_QUESTION_CHARS}
          rows={2}
          placeholder="Ask about a company's 10-K or 10-Q filings…"
          className="block w-full resize-none rounded-xl border border-slate-300 bg-white px-3 py-2.5 text-base shadow-sm placeholder:text-slate-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 focus:outline-none"
        />
        {remaining <= 100 && (
          <span className="pointer-events-none absolute right-2 bottom-1 text-xs text-slate-500">
            {remaining}
          </span>
        )}
      </div>
      <button
        type="submit"
        disabled={!canSend}
        className="h-11 shrink-0 rounded-xl bg-blue-600 px-4 font-medium text-white hover:bg-blue-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        Send
      </button>
    </form>
  );
}
