import time

from conftest import DELAY

from app import pipeline
from app.logger import read_log
from app.steps.standalone import Message


def steps(tmp_path, run_id: str) -> list[str]:
    records = [r for f in tmp_path.glob("*.jsonl") for r in read_log(f)]
    return sorted(r["step"] for r in records if r["run_id"] == run_id)


def test_metric_question_searches_and_calls_tool_in_parallel(fakes, tmp_path):
    t0 = time.perf_counter()
    r = pipeline.answer("metric Apple revenue 2024")
    elapsed = time.perf_counter() - t0

    assert elapsed < 2 * DELAY  # search and tool overlapped
    assert r.route == "metric" and r.behavior == "answer"
    assert r.tool and r.tool.chart == "table" and r.tool.rows[0].ticker == "AAPL"
    assert [c.kind for c in r.citations] == ["filing", "metric"]
    assert r.citations[0].url.endswith("doc.htm")
    assert r.citations[1].url.startswith("https://www.sec.gov/Archives/edgar/data/")
    assert set(r.ms) == {"route", "search", "tool", "generate", "total"}
    assert steps(tmp_path, r.run_id) == ["check", "generate", "route", "search", "tool"]


def test_narrative_question_searches_only(fakes, tmp_path):
    r = pipeline.answer("narrative What risks does Apple describe?")
    assert r.tool is None and [c.kind for c in r.citations] == ["filing"]
    assert steps(tmp_path, r.run_id) == ["check", "generate", "route", "search"]


def test_clarify_and_refuse_get_no_sources(fakes, tmp_path):
    for question, behavior in (("vague What about it?", "clarify"),
                               ("advice Should I buy?", "refuse")):  # fmt: skip
        r = pipeline.answer(question)
        assert r.behavior == behavior and r.citations == [] and r.tool is None
        assert r.unsupported == []  # not checked
        assert fakes.generated[-1]["chunks"] == []
        assert steps(tmp_path, r.run_id) == ["generate", "route"]


def test_answer_with_a_figure_its_source_lacks_is_flagged(fakes, monkeypatch):
    real_generate = pipeline.generate

    def generate(row, model):
        out = real_generate(row, model)  # the fake in conftest
        return {**out, "answer": "Revenue was $412 billion [1]."}

    monkeypatch.setattr(pipeline, "generate", generate)
    r = pipeline.answer("narrative Apple revenue 2024")
    assert r.behavior == "answer"  # flagged, not blocked
    assert [u.numbers for u in r.unsupported] == [["$412 billion"]]
    assert r.unsupported[0].cited == [1]


def test_decline_that_cites_a_source_is_checked(fakes, monkeypatch, tmp_path):
    real_generate = pipeline.generate

    def generate(row, model):
        out = real_generate(row, model)  # the fake in conftest: cites chunk [1]
        answer = "The filings don't cover that; total revenue was $412 billion [1]."
        return {**out, "behavior": "refuse", "answer": answer}

    monkeypatch.setattr(pipeline, "generate", generate)
    r = pipeline.answer("narrative Apple revenue 2024")
    assert r.behavior == "refuse" and r.citations
    assert [u.numbers for u in r.unsupported] == [["$412 billion"]]
    assert "check" in steps(tmp_path, r.run_id)


def test_first_question_is_not_rewritten(fakes):
    r = pipeline.answer("metric Apple revenue 2024", [])
    assert fakes.rewrites == 0
    assert r.standalone_question == r.question == "metric Apple revenue 2024"
    assert "standalone" not in r.ms


def test_follow_up_is_rewritten_and_used_downstream(fakes, tmp_path):
    history = [
        Message(role="user", content="What was Apple's revenue in fiscal 2023?"),
        Message(role="assistant", content="It was $383 billion [21]."),
    ]
    r = pipeline.answer("What about 2022?", history)
    assert fakes.rewrites == 1
    assert r.question == "What about 2022?"
    assert r.standalone_question == "metric Apple revenue fiscal 2022"
    assert fakes.routed[-1] == fakes.generated[-1]["query"] == r.standalone_question
    assert "standalone" in r.ms
    assert "standalone" in steps(tmp_path, r.run_id)
