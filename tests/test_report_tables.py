"""Spec 10 tasks 3–4: the four table renderers.

The assertions here are mostly *honesty* assertions, not formatting ones — that the baseline column
exists, that a regression is printed, that a `null` sub-metric is printed with its reason, and that
a synthetic stage never renders a number. No model is loaded, so this file is not `model_heavy`.
"""

from __future__ import annotations

import pytest

from gdp.config import DRIVELM_CATEGORIES
from gdp.paths import repo_root
from gdp.report import tables
from gdp.report.snapshot import SYNTHETIC_REASON, build_snapshot

FIXTURE_RUNS = repo_root() / "tests" / "fixtures" / "report" / "runs"


@pytest.fixture(scope="module")
def blocks():
    return tables.render_all(build_snapshot(runs_root=FIXTURE_RUNS))


def _text(lines):
    return "\n".join(lines)


# --------------------------------------------------------------------------- pending / synthetic


def _pending_entry(**overrides):
    entry = {
        "status": "pending",
        "blocker": "spec 03 cluster fine-tuning run",
        "reason": "no artifact yet",
        "source": None,
        "data": {},
    }
    entry.update(overrides)
    return entry


@pytest.mark.parametrize(
    "renderer",
    [
        tables.render_stage1_map,
        tables.render_grounding,
        tables.render_deployment,
        tables.render_stage2_vqa,
    ],
)
def test_pending_names_its_blocker_and_prints_no_number(renderer):
    text = _text(renderer(_pending_entry()))
    assert "Not yet measured" in text
    assert "spec 03 cluster fine-tuning run" in text
    assert "|" not in text, "a pending block must not render a table row"


@pytest.mark.parametrize(
    "renderer",
    [
        tables.render_stage1_map,
        tables.render_grounding,
        tables.render_deployment,
        tables.render_stage2_vqa,
    ],
)
def test_synthetic_renders_like_pending_and_says_why(renderer):
    """H7's load-bearing case: an artifact exists, and is still not a result."""
    entry = _pending_entry(
        status="synthetic",
        reason=SYNTHETIC_REASON,
        source="runs/05-deploy/20260725-163907/metrics.json",
        data={"metrics": {"variants": {"int8-onnx": {"map": 0.9}}}},
    )
    text = _text(renderer(entry))
    assert "Not yet measured" in text
    assert SYNTHETIC_REASON in text
    assert "0.9" not in text


# --------------------------------------------------------------------------- Stage 1 · mAP


def test_stage1_has_both_columns_and_the_delta(blocks):
    text = _text(blocks["stage1-map"])
    assert "Zero-shot (baseline)" in text and "Fine-tuned" in text
    assert "| mAP@[.50:.95] | 0.211 | 0.348 | +0.137 |" in text
    assert "| mAP@0.50 | 0.372 | 0.561 | +0.189 |" in text


def test_stage1_keeps_every_class_including_the_one_that_got_worse(blocks):
    text = _text(blocks["stage1-map"])
    for cls in ("bus", "car", "pedestrian", "traffic light"):
        assert f"`{cls}`" in text
    assert "| `bus` | 0.301 | 0.287 | -0.014 |" in text


def test_stage1_states_its_regressions(blocks):
    assert "**Regressions:** `bus`" in _text(blocks["stage1-map"])


def test_stage1_footline_carries_split_count_and_dates(blocks):
    text = _text(blocks["stage1-map"])
    assert "runs/03-finetune/20260801-101500/comparison.json" in text
    assert "split `val`" in text and "10000 images" in text
    assert "2026-08-01T10:15:00+00:00" in text


def test_no_regressions_says_so_rather_than_omitting_the_line():
    assert "none" in tables._regression_line([], unit="class")


# --------------------------------------------------------------------------- grounding


def test_grounding_has_both_columns_positives_and_negatives(blocks):
    text = _text(blocks["grounding"])
    assert "| Overall | 0.410 | 0.583 | +0.172 |" in text
    assert "Positive phrases" in text and "Negative phrases" in text


def test_grounding_per_type_includes_the_regressed_type(blocks):
    text = _text(blocks["grounding"])
    assert "| `colour` | 0.500 | 0.450 | -0.050 |" in text
    assert "**Regressions:** `colour`" in text


def test_grounding_caveat_is_copied_verbatim(blocks):
    """H9's sanctioned exception is only sanctioned *because* the label travels with the number."""
    text = _text(blocks["grounding"])
    assert "Small, self-built grounding evaluation set" in text
    assert "data/grounding_eval/construction.md" in text
    assert "Self-built evaluation set" in text


def test_grounding_footline_carries_phrase_count_and_hash(blocks):
    text = _text(blocks["grounding"])
    assert "212 frozen phrases" in text
    assert "b7f1c0d9e2a3" in text


# --------------------------------------------------------------------------- deployment


def test_deployment_rows_are_in_spec05_variant_order(blocks):
    rows = [ln for ln in blocks["deployment"] if ln.startswith("| `")]
    assert [r.split("`")[1] for r in rows] == ["fp32-pt", "fp32-onnx", "int8-onnx"]


def test_deployment_pairs_every_latency_with_its_accuracy(blocks):
    """H8: a speed number with no mAP beside it is half a result."""
    text = _text(blocks["deployment"])
    assert "| `int8-onnx` | 0.330 | 0.540 | 0.244 | 0.298 | 0.341 | 4.09 |" in text


def test_deployment_states_the_fixed_prompt_export_tradeoff(blocks):
    text = _text(blocks["deployment"])
    assert "fixed-vocabulary, not open-vocabulary" in text
    assert "`fixed_prompt: true`" in text


def test_deployment_footline_carries_hardware_and_iterations(blocks):
    text = _text(blocks["deployment"])
    assert "onnxruntime 1.28.0" in text
    assert "200 timed iterations" in text
    assert "latency.json" in text


# --------------------------------------------------------------------------- Stage 2 · VQA


def test_stage2_has_the_base_vlm_baseline_column(blocks):
    text = _text(blocks["stage2-vqa"])
    assert "Base VLM (baseline)" in text
    assert "| Overall | 0.431 | 0.571 | +0.140 |" in text


def test_stage2_categories_are_in_official_drivelm_order(blocks):
    rows = [ln for ln in blocks["stage2-vqa"] if ln.startswith("| `") and "|" in ln]
    ordered = [r.split("`")[1] for r in rows]
    categories = [c for c in ordered if c in DRIVELM_CATEGORIES]
    assert categories == list(DRIVELM_CATEGORIES)


def test_stage2_keeps_the_regressed_category_and_the_unscored_one(blocks):
    text = _text(blocks["stage2-vqa"])
    assert "| `behavior` | 0.411 | 0.390 | -0.021 |" in text
    assert "**Regressions:** `behavior`" in text
    assert f"| `prediction` | {tables.NOT_SCORED} |" in text
    assert tables.NOT_SCORED_NOTE in text


def test_stage2_says_planning_and_behavior_are_the_risk_result(blocks):
    text = _text(blocks["stage2-vqa"])
    assert "risk-reasoning result" in text
    assert "no separate risk engine" in text


def test_stage2_reports_final_score_null_with_both_omitted_submetrics(blocks):
    """Decision 7 — dropping the uncomputable rows is the exact reflex the contract names."""
    text = _text(blocks["stage2-vqa"])
    assert "`final_score`: not computed" in text
    assert "`chatgpt`" in text and "`match`" in text
    assert "paid, non-deterministic OpenAI" in text
    assert "third_party/drivelm/PROVENANCE.md" in text


def test_stage2_hallucination_is_labelled_a_diagnostic(blocks):
    text = _text(blocks["stage2-vqa"])
    assert "Ungrounded object-tag rate" in text
    assert "self-defined diagnostic, not part of DriveLM's official metric" in text


def test_stage2_states_it_is_not_the_leaderboard_split(blocks):
    text = _text(blocks["stage2-vqa"])
    assert "Not comparable to the DriveLM leaderboard" in text
    assert "NOT the DriveLM leaderboard split" in text
    assert "nuscenes.utils.splits" in text


# --------------------------------------------------------------------------- render_all


def test_render_all_returns_exactly_the_declared_block_names(blocks):
    assert set(blocks) == set(tables.BLOCK_NAMES)


def test_render_all_raises_on_a_snapshot_missing_a_stage():
    with pytest.raises(KeyError, match="stage1_detection"):
        tables.render_all({"stages": {}})
