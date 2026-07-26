"""The H-guard test file (spec 09 task 3): exact constant text, that `render_report` tags every
line, that an imperative command raises, and that `src/gdp/demo/` contains none of
`claims-auditor`'s H3/H9 danger tokens. No model is loaded, so this file is intentionally NOT
`model_heavy` (CLAUDE.md §4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gdp.demo import honesty
from gdp.demo.report import Attribution, ReportLine, ReportSection, render_report
from gdp.paths import repo_root

DEMO_SRC_DIR = repo_root() / "src" / "gdp" / "demo"

# H3 danger words (CLAUDE.md's "edge-deployed never means deployed in a vehicle").
H3_DANGER_WORDS = ("in-vehicle", "production", "real-time", "deployed in")

# H9 danger tokens — the risk overlay must emit zero numbers, scores, or levels (design decision 8).
H9_DANGER_TOKENS = ("risk level", "risk score", "hazard probability")


def test_separate_models_caption_exact_text():
    assert honesty.SEPARATE_MODELS == (
        "Two separate models. The detector does not feed the VLM. "
        "Stage 1 = Grounding-DINO (boxes). Stage 2 = Qwen2.5-VL (text)."
    )


def test_report_caveat_exact_text():
    assert honesty.REPORT_CAVEAT == (
        "⚠ Qualitative demo. Composed from two separate models. Not a measured result."
    )


def test_heuristic_badge_exact_text():
    assert honesty.HEURISTIC_BADGE == "heuristic illustration — no accuracy claim (H9)"


def test_unmeasured_query_exact_text():
    assert honesty.UNMEASURED_QUERY == (
        "demo, not a measured result — this phrase has no ground truth (H7)"
    )


def test_synthetic_scene_exact_text():
    assert honesty.SYNTHETIC_SCENE == "synthetic fixture — NOT real BDD100K"


def test_zeroshot_weights_exact_text():
    assert honesty.ZEROSHOT_WEIGHTS == "zero-shot checkpoint — NOT the fine-tuned model (H8)"


def test_base_vlm_weights_exact_text():
    assert honesty.BASE_VLM_WEIGHTS == (
        "base VLM, no LoRA adapter loaded — NOT the fine-tuned model (H8)"
    )


def test_render_report_tags_every_emitted_line():
    section = ReportSection(
        "Scene summary",
        (
            ReportLine("2 cars detected.", Attribution.DETECTOR),
            ReportLine("A cyclist is ahead.", Attribution.VLM),
            ReportLine("1 car is ahead of the ego vehicle.", Attribution.HEURISTIC),
        ),
    )
    rendered = render_report([section])
    for line in rendered.splitlines():
        if line.startswith("##") or not line.strip():
            continue
        assert line.rstrip().endswith("]")


def test_report_line_rejects_brake_now():
    with pytest.raises(ValueError):
        ReportLine("Brake now.", Attribution.VLM)


def _demo_source_files() -> list[Path]:
    return sorted(DEMO_SRC_DIR.rglob("*.py"))


@pytest.mark.parametrize("word", H3_DANGER_WORDS)
def test_no_h3_danger_words_in_demo_source(word):
    for path in _demo_source_files():
        text = path.read_text().lower()
        assert word not in text, f"{word!r} found in {path}"


@pytest.mark.parametrize("token", H9_DANGER_TOKENS)
def test_no_h9_danger_tokens_in_demo_source(token):
    for path in _demo_source_files():
        text = path.read_text().lower()
        assert token not in text, f"{token!r} found in {path}"


def test_no_percent_adjacent_to_risk_in_demo_source():
    """`claims-auditor`'s exact grep target: a '%' near the word 'risk'."""
    for path in _demo_source_files():
        text = path.read_text().lower()
        for idx, ch in enumerate(text):
            if ch != "%":
                continue
            window = text[max(0, idx - 20) : idx + 20]
            assert "risk" not in window, f"'%' adjacent to 'risk' found in {path}: {window!r}"
