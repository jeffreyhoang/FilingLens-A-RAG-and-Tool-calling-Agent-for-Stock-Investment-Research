const EXAMPLES = [
  "How does Apple describe the main risks to its business in its fiscal 2024 10-K?",
  "What was NVIDIA's revenue in fiscal 2025?",
  "Compare revenue for Apple, Microsoft and Alphabet in fiscal 2024.",
  "Show Tesla's net income from fiscal 2019 to 2024.",
  "How did Microsoft's operating income change in fiscal 2024, and what drove it?",
];

const COMPANIES = "Apple, Microsoft, Alphabet, Amazon, Netflix, NVIDIA, Tesla, Cisco, Oracle and Adobe";

export function EmptyState({ onPick }: { onPick: (question: string) => void }) {
  return (
    <div className="mx-auto max-w-xl py-8 text-center sm:py-14">
      <h2 className="text-xl font-semibold text-slate-900">Ask about a company's filings</h2>
      <p className="mt-2 text-sm text-slate-600">
        Answers come from 10-K and 10-Q filings for {COMPANIES}, fiscal 2016 onward, with
        citations to the filing.
      </p>
      <ul className="mt-6 space-y-2 text-left">
        {EXAMPLES.map((q) => (
          <li key={q}>
            <button
              type="button"
              onClick={() => onPick(q)}
              className="w-full rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-left text-sm text-slate-700 shadow-sm hover:border-blue-300 hover:bg-blue-50 focus-visible:outline-2 focus-visible:outline-blue-600"
            >
              {q}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
