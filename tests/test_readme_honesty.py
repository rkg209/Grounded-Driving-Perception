"""Spec 10 task 11: the README's own honesty guards.

Mirrors `tests/test_demo_honesty.py`'s idiom (danger-word greps + exact constants), applied to the
one file a reviewer actually reads. Two properties are asserted:

1. **Traceability** — no metric-shaped number appears in hand-written prose. Numbers live inside
   generated blocks, which come from `docs/metrics_snapshot.json`, which comes from `runs/`.
2. **Definition locality** — `claims-auditor.md`'s danger words may appear only in the sections
   that define them (`Honest scope`, `Limitations`, the deployment section, the charter quote).
   "Deployed" is fine when its next clause says what it means; it is an H3 violation anywhere else.

No model is loaded, so this file is not `model_heavy`.
"""

from __future__ import annotations

import re

import pytest

from gdp.paths import repo_root
from gdp.report.inject import marker_spans
from gdp.report.tables import BLOCK_NAMES

README = repo_root() / "README.md"

# Sections where a term is *defined* rather than claimed. A danger word outside these is an
# unqualified claim.
DEFINING_SECTIONS = (
    "Honest scope — what this is NOT",
    "Limitations & next steps",
    "Deployment — accuracy vs latency (M4 laptop as the edge target)",
    "Interview one-liners",
    'About "risk assessment"',
)

H3_DANGER_WORDS = ("deployed", "production", "in-car", "real-time")
H5_DANGER_WORDS = ("tracker", "trajectory prediction", "planner", "control command")
H1_DANGER_WORDS = ("from scratch",)

# Never acceptable, in any section.
SUPERLATIVES = ("state-of-the-art", "cutting-edge", "world-class", "unprecedented", "best-in-class")


def _lines() -> list[str]:
    return README.read_text().splitlines()


def _generated_line_numbers() -> set[int]:
    """Line indices inside (and including) any generated block — owned by the renderer."""
    text = README.read_text()
    owned: set[int] = set()
    for _, (start, end) in marker_spans(text).items():
        owned.update(range(start, end + 1))
    return owned


def _prose_lines() -> list[tuple[int, str]]:
    generated = _generated_line_numbers()
    return [(i, line) for i, line in enumerate(_lines()) if i not in generated]


def _section_of(index: int) -> str:
    section = "(preamble)"
    for i, line in enumerate(_lines()):
        if i > index:
            break
        if line.startswith("#"):
            section = line.lstrip("#").strip()
    return section


# --------------------------------------------------------------------------- traceability


def test_no_metric_shaped_number_in_hand_written_prose():
    """A number beside `mAP`, `%`, or `accuracy` in prose is a number somebody typed. Every real
    one belongs in a generated block, beside its baseline and its provenance (H8)."""
    pattern = re.compile(r"(\d[\d.]*\s*%|\d[\d.]*\s*mAP|mAP\s*[:=]?\s*\d|accuracy\s*(of|:)\s*\d)")
    offenders = [(i + 1, line) for i, line in _prose_lines() if pattern.search(line)]
    assert offenders == [], f"hand-typed metric in README prose: {offenders}"


def test_readme_contains_every_generated_block():
    present = set(marker_spans(README.read_text()))
    assert set(BLOCK_NAMES) <= present


def test_readme_points_at_the_snapshot_as_the_source_of_its_numbers():
    text = README.read_text()
    assert "docs/metrics_snapshot.json" in text
    assert "uv run gdp report render" in text


def test_readme_says_the_fixture_stages_are_not_results():
    """The one thing a reader could most easily misread today: artifacts exist, results do not."""
    text = README.read_text()
    assert "not a result" in text
    assert "tests/fixtures/" in text


# --------------------------------------------------------------------------- danger words


@pytest.mark.parametrize("word", H3_DANGER_WORDS + H5_DANGER_WORDS + H1_DANGER_WORDS)
def test_danger_words_only_appear_where_they_are_defined(word):
    offenders = [
        (i + 1, _section_of(i), line)
        for i, line in _prose_lines()
        if re.search(rf"\b{re.escape(word)}\b", line.lower())
        and _section_of(i) not in DEFINING_SECTIONS
    ]
    assert offenders == [], f"{word!r} used outside a defining section: {offenders}"


@pytest.mark.parametrize("word", SUPERLATIVES)
def test_no_superlatives_anywhere(word):
    assert word not in README.read_text().lower()


def test_edge_deployed_always_carries_its_definition():
    """H3's exact requirement: the phrase never travels without quantized + ONNX + benchmarked."""
    text = README.read_text()
    for match in re.finditer(r"[Ee]dge-deployed", text):
        window = text[match.start() : match.start() + 400]
        assert "quantized" in window and "latency-benchmarked" in window, (
            f"'edge-deployed' at offset {match.start()} without its definition nearby"
        )


def test_joint_model_is_only_ever_denied():
    """H6. The phrase is allowed — "Not one joint model" is the honest sentence — but only when
    the words immediately before it are a denial."""
    text = README.read_text().lower()
    for phrase in ("joint model", "one model", "end-to-end system"):
        for match in re.finditer(re.escape(phrase), text):
            before = text[max(0, match.start() - 40) : match.start()]
            assert re.search(r"\b(not|no|never|separate)\b", before), (
                f"{phrase!r} appears without a denial in front of it (H6): "
                f"{text[max(0, match.start() - 60) : match.start() + 40]!r}"
            )


def test_risk_is_never_given_a_number():
    """`claims-auditor.md`: any number adjacent to 'risk' is an H9 violation until proven to come
    from DriveLM's official scored categories."""
    text = README.read_text().lower()
    for match in re.finditer(r"risk", text):
        window = text[max(0, match.start() - 60) : match.start() + 60]
        assert "%" not in window, f"'%' near 'risk': {window!r}"
