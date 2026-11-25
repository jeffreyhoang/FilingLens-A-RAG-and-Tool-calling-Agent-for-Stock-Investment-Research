from dataclasses import dataclass, field

import pytest

from app.steps.citecheck import check, figures, sentences


@dataclass
class Src:
    n: int
    text: str
    label: str = "AAPL 10-K, fiscal 2024 (period ending 2024-09-28), Item 7: MD&A"
    cite: dict = field(default_factory=lambda: {"kind": "filing"})


def metric(n: int, text: str, year: int) -> Src:
    return Src(n, text, f"AAPL revenue, fiscal {year}", {"kind": "metric"})


TABLE = Src(1, "Total net sales (in millions) 391,035 383,285 Services 96,169 85,200")
TEXT = Src(
    2,
    "Services revenue grew 13% to $96.2 billion, including advertising revenue of "
    "$31 billion.",
)


def flagged(answer: str, srcs: list[Src]) -> list[list[str]]:
    return [u.numbers for u in check(answer, srcs)]


@pytest.mark.parametrize(
    "answer",
    [
        "Apple's revenue was $391 billion in fiscal 2024 [1].",  # table in millions
        "Apple's revenue was $391.0 billion [1].",
        "Services revenue was $96.2 billion, up 13% [2].",  # stated directly
        "Revenue rose $7.75 billion [1].",  # difference: 391,035 - 383,285
        "Revenue grew about 2.0% [1].",  # percent change
        "Services grew 12.9% [1].",  # percent change, more digits
        "Apple has 2 segments and earned $391 billion [1].",  # small count skipped
    ],
)
def test_supported(answer):
    assert flagged(answer, [TABLE, TEXT]) == []


@pytest.mark.parametrize(
    ("answer", "numbers"),
    [
        ("Apple's revenue was $412 billion [1].", ["$412 billion"]),  # made up
        ("Apple's revenue was $391.5 billion [1].", ["$391.5 billion"]),  # too precise
        (
            "Advertising revenue was $31 billion [1].",
            ["$31 billion"],
        ),  # in [2], not [1]
        ("Revenue grew 9% [1].", ["9%"]),
        ("Apple's revenue was $391 billion in fiscal 2021 [1].", ["2021"]),  # year
    ],
)
def test_unsupported(answer, numbers):
    assert flagged(answer, [TABLE, TEXT]) == [numbers]


def test_metric_rows_support_values_and_changes_between_them():
    srcs = [
        metric(1, "revenue: $383,285,000,000 ($383.29 billion)", 2023),
        metric(2, "revenue: $391,035,000,000 ($391.04 billion)", 2024),
    ]
    answer = (
        "Revenue was $383 billion in fiscal 2023 [1] and $391 billion in fiscal "
        "2024 [2], up 2.0% [1][2]."
    )
    assert flagged(answer, srcs) == []
    assert flagged("Revenue rose 1.02x [1][2].", srcs) == []
    assert flagged("Revenue was $391 billion in fiscal 2024 [1].", srcs) == [
        ["$391 billion", "2024"]
    ]  # cites the 2023 row: the wrong period


def test_uncited_sentence_is_checked_against_the_answers_sources():
    answer = (
        "Revenue was $391 billion [1]. That was the highest ever.\nIt was $500 billion."
    )
    result = check(answer, [TABLE, TEXT])
    assert [(u.sentence, u.numbers, u.cited) for u in result] == [
        ("It was $500 billion.", ["$500 billion"], [1])
    ]


def test_markers_forms_quarters_and_sections_are_not_figures():
    text = "See Item 7 and Note 12 of the 10-K for Q3 [4][5]; sales were $1.2 billion."
    assert [f.text for f in figures(text)] == ["$1.2 billion"]


def test_figure_values_and_kinds():
    got = {f.text: (f.value, f.digits, f.kind) for f in figures(
        "$4.4 billion, 14%, 12.0x, 2024, 3,500, 301.6 million, $6.08"
    )}  # fmt: skip
    assert got == {
        "$4.4 billion": (pytest.approx(4.4e9), 2, "amount"),
        "14%": (pytest.approx(14), 2, "percent"),
        "12.0x": (pytest.approx(12.0), 3, "multiple"),
        "2024": (pytest.approx(2024), 4, "year"),
        "3,500": (pytest.approx(3500), 2, "amount"),
        "301.6 million": (pytest.approx(301.6e6), 4, "amount"),
        "$6.08": (pytest.approx(6.08), 3, "amount"),
    }


def test_sentences_keep_the_answer_text():
    answer = "Revenue was $4.4 billion [1]. Margins rose [2].\n- Item one"
    assert sentences(answer) == [
        "Revenue was $4.4 billion [1].",
        "Margins rose [2].",
        "- Item one",
    ]
    assert all(s in answer for s in sentences(answer))
