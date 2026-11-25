import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  chat,
  health,
  MAX_HISTORY,
  type ChatResponse,
  type HistoryMessage,
} from "./api.ts";
import { Composer } from "./components/Composer.tsx";
import { EmptyState } from "./components/EmptyState.tsx";
import { AssistantMessage, ErrorMessage, Loading, UserMessage } from "./components/Messages.tsx";

// The conversation lives only in memory: a page refresh clears it.
type Turn =
  | { id: number; kind: "user"; text: string }
  | { id: number; kind: "assistant"; response: ChatResponse }
  | { id: number; kind: "error"; text: string };

// A turn before it gets its id (Omit applied to each member of the union)
type NewTurn = Turn extends infer T ? (T extends Turn ? Omit<T, "id"> : never) : never;

type Status = "checking" | "online" | "offline";

/** Earlier user questions and assistant replies, as the API expects them. */
function toHistory(turns: Turn[]): HistoryMessage[] {
  const history: HistoryMessage[] = [];
  for (const t of turns) {
    if (t.kind === "user") history.push({ role: "user", content: t.text });
    if (t.kind === "assistant") history.push({ role: "assistant", content: t.response.answer });
  }
  return history.slice(-MAX_HISTORY);
}

function StatusDot({ status }: { status: Status }) {
  const styles = {
    checking: ["bg-slate-300", "Connecting…"],
    online: ["bg-emerald-500", "Backend connected"],
    offline: ["bg-red-500", "Backend offline"],
  }[status];
  return (
    <span className="flex items-center gap-1.5 text-xs text-slate-500">
      <span className={`h-2 w-2 rounded-full ${styles[0]}`} aria-hidden />
      {styles[1]}
    </span>
  );
}

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<Status>("checking");
  const nextId = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ac = new AbortController();
    health(ac.signal).then((ok) => {
      if (!ac.signal.aborted) setStatus(ok ? "online" : "offline");
    });
    return () => ac.abort();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, loading]);

  function add(turn: NewTurn) {
    const id = nextId.current++;
    setTurns((prev) => [...prev, { ...turn, id } as Turn]);
  }

  async function send(question: string) {
    if (loading) return;
    const history = toHistory(turns);
    add({ kind: "user", text: question });
    setLoading(true);
    const ac = new AbortController();
    controller.current = ac;
    try {
      const response = await chat(question, history, ac.signal);
      add({ kind: "assistant", response });
      setStatus("online");
    } catch (err) {
      if (ac.signal.aborted) return; // "New chat" during the request
      const offline = err instanceof ApiError && err.offline;
      if (offline) setStatus("offline");
      else setStatus("online");
      add({
        kind: "error",
        text: err instanceof ApiError ? err.message : "Something went wrong. Try again.",
      });
    } finally {
      if (controller.current === ac) {
        controller.current = null;
        setLoading(false);
      }
    }
  }

  function newChat() {
    controller.current?.abort();
    controller.current = null;
    setLoading(false);
    setTurns([]);
  }

  return (
    <div className="flex h-dvh flex-col">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-3xl flex-wrap items-center justify-between gap-2 px-4 py-3">
          <div className="flex items-baseline gap-3">
            <h1 className="text-lg font-semibold tracking-tight">FilingLens</h1>
            <StatusDot status={status} />
          </div>
          <button
            type="button"
            onClick={newChat}
            disabled={turns.length === 0 && !loading}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-blue-600 disabled:cursor-not-allowed disabled:opacity-50"
          >
            New chat
          </button>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl space-y-4 px-4 py-6" aria-live="polite">
          {turns.length === 0 && !loading ? (
            <EmptyState onPick={send} />
          ) : (
            turns.map((t) => {
              if (t.kind === "user") return <UserMessage key={t.id} text={t.text} />;
              if (t.kind === "assistant") return <AssistantMessage key={t.id} response={t.response} />;
              return <ErrorMessage key={t.id} text={t.text} />;
            })
          )}
          {loading && <Loading />}
          <div ref={endRef} />
        </div>
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-3xl px-4 py-3">
          <Composer disabled={loading} onSend={send} />
          <p className="mt-2 text-center text-xs text-slate-500">
            Research only, not financial advice. Answers come from SEC filings and may
            contain errors; check the cited filing.
          </p>
        </div>
      </footer>
    </div>
  );
}
