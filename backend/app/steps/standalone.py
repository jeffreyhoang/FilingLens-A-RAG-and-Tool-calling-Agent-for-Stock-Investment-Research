"""Standalone: one LLM call rewrites a follow-up so it can be answered without the
conversation.

- Without chat history there is no call; the question is used as typed.

Run with:
    uv run python -m app.steps.standalone
"""

import argparse
import json
import time
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

MODEL = "gpt-5.4-mini"
MAX_TOKENS = 1024
MAX_MESSAGES = 6  # earlier messages kept: the last three exchanges
MAX_CHARS = 1500  # per earlier message; long answers are cut


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10_000)


class Standalone(BaseModel):
    question: str


PROMPT = """\
You prepare follow-up questions for a research assistant that answers questions \
from the 10-K and 10-Q filings of 10 large tech companies. You get the earlier \
conversation and the user's latest question. Rewrite the latest question so it \
can be understood on its own, without the conversation.

Rules:
- Fill in what the latest question leaves out but the conversation makes clear: \
the company, the fiscal year or quarter, and the figure or topic ("they", "that \
year", "what about 2022?", "and Microsoft?").
- When the user answers a clarifying question (the assistant asked "Which \
company do you mean?"), combine the answer with the question it clarifies.
- If the latest question already stands on its own, or changes the subject, \
return it unchanged.
- If the conversation doesn't make clear what the question refers to, return it \
unchanged; the assistant will ask.
- Keep the user's wording and intent otherwise. Don't answer it, don't add facts \
or years the conversation doesn't contain, and keep relative dates ("last year") \
as they are.

Examples:
- user: "What was Netflix's revenue in fiscal 2021?" / assistant: "Netflix's \
revenue was ... [1]." / latest: "And in 2019?" -> "What was Netflix's revenue in \
fiscal 2019?"
- user: "What supply chain risks does Cisco describe?" / assistant: "Cisco \
describes component shortages ... [2]." / latest: "Did they hurt its gross margin \
last year?" -> "Did supply chain issues hurt Cisco's gross margin last year?"
- user: "What did the company say about layoffs?" / assistant: "Which company do \
you mean?" / latest: "Microsoft" -> "What did Microsoft say about layoffs?"
- user: "What was Amazon's net income in 2020?" / assistant: "... [1]." / latest: \
"What was Tesla's R&D expense in 2023?" -> unchanged

Today's date: {today}."""

# Hand-written conversations: (earlier messages, latest question)
SAMPLES = [
    (
        [
            ("user", "What was Apple's revenue in fiscal 2023?"),
            ("assistant", "Apple's revenue in fiscal 2023 was $383 billion [21]."),
        ],
        "What about 2022?",
    ),
    (
        [
            ("user", "How does Oracle describe competition in cloud infrastructure?"),
            ("assistant", "Oracle says it competes with large cloud providers [3]."),
        ],
        "How much did they spend on R&D in fiscal 2024?",
    ),
    (
        [
            ("user", "What was the operating income in the second quarter of 2021?"),
            ("assistant", "Which company do you mean?"),
        ],
        "Adobe",
    ),
    (
        [
            ("user", "What was NVIDIA's diluted EPS in fiscal 2024?"),
            ("assistant", "NVIDIA's diluted EPS in fiscal 2024 was $11.93 [21]."),
        ],
        "Compare that with AMD.",
    ),
]


def conversation(history: list[Message], question: str) -> str:
    lines = [f"{m.role}: {m.content[:MAX_CHARS]}" for m in history[-MAX_MESSAGES:]]
    return "Conversation:\n" + "\n".join(lines) + f"\n\nLatest question: {question}"


def standalone(
    question: str, history: list[Message], today: date | None = None
) -> tuple[str, dict]:
    """The question rewritten to stand on its own, and {ms, input_tokens,
    output_tokens}. With no history, the question unchanged and no call."""
    if not history:
        return question, {"ms": 0, "input_tokens": 0, "output_tokens": 0}
    from app.retrieval.embed import openai_client

    t0 = time.perf_counter()
    resp = openai_client().responses.parse(
        model=MODEL,
        instructions=PROMPT.format(today=(today or date.today()).isoformat()),
        input=conversation(history, question),
        text_format=Standalone,
        reasoning={"effort": "low"},  # matches the router; keeps the call quick
        max_output_tokens=MAX_TOKENS,
    )
    if resp.output_parsed is None:
        raise RuntimeError(f"{MODEL} returned no standalone question for {question!r}")
    stats = {
        "ms": round((time.perf_counter() - t0) * 1000),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    return resp.output_parsed.question.strip() or question, stats


def main() -> None:
    from app.steps.router import route

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    for turns, question in SAMPLES:
        history = [Message(role=r, content=c) for r, c in turns]
        rewritten, stats = standalone(question, history)
        routed, _ = route(rewritten, "gpt-5.4-mini")
        print(f"\n{turns[0][1]}  ->  {question!r}")
        print(f"  standalone: {rewritten}")
        print(f"  {routed.route} {json.dumps(routed.filters.model_dump())}  {stats}")


if __name__ == "__main__":
    main()
