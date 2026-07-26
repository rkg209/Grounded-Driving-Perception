"""Spec 08 task 5: `gdp.vqa.score` — overall + per-category scoring, the omitted-submetric ->
`final_score: null` gate, and zero-item subsets reported as `null` with a reason, never `0.0`.

No model dependency. The round-trip through the real vendored scorer needs
`language_evaluation`/`openai` installed (see `tests/test_vqa_official.py`); these tests skip
themselves with a clear reason when they aren't, mirroring that file's pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gdp.config import VQAEvalConfig, load_config
from gdp.data.drivelm import convert_drivelm
from gdp.data.drivelm_stats import CAVEAT, SPLIT_PROVENANCE
from gdp.data.splits import load_official_splits, load_scene_meta
from gdp.vqa.official import OfficialScorerUnavailable
from gdp.vqa.score import (
    annotate_per_item,
    build_vqa_metrics,
    score_predictions,
    write_per_item,
    write_vqa_metrics,
)


@pytest.fixture(scope="module")
def val_records():
    cfg = load_config("configs/default.yaml").drivelm
    raw = json.loads(cfg.annotations_path().read_text())
    scene_meta = load_scene_meta(cfg.scene_meta_path())
    splits = load_official_splits()
    _, val, _ = convert_drivelm(
        raw,
        scene_meta=scene_meta,
        splits=splits,
        images_root=cfg.nuscenes_root_path(),
        categories=tuple(cfg.categories),
        is_synthetic=True,
    )
    return val


@pytest.fixture(scope="module")
def base_predictions():
    path = Path("tests/fixtures/vqa_predictions/base_mini.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


def _skip_if_scorer_unavailable(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except OfficialScorerUnavailable as exc:
        pytest.skip(f"vendored scorer deps not installed: {exc}")


def test_score_predictions_raises_on_incomplete_split(val_records, base_predictions):
    with pytest.raises(ValueError, match="missing qa_id"):
        _skip_if_scorer_unavailable(
            score_predictions,
            base_predictions[:-1],
            val_records,
            cfg=__import__("gdp.config", fromlist=["VQAEvalConfig"]).VQAEvalConfig(),
        )


def test_score_predictions_overall_and_per_category_shape(val_records, base_predictions):
    scored = _skip_if_scorer_unavailable(
        score_predictions, base_predictions, val_records, cfg=VQAEvalConfig()
    )
    assert set(scored) == {"overall", "per_category"}
    assert set(scored["per_category"]) == {"perception", "prediction", "planning", "behavior"}

    overall = scored["overall"]
    assert overall["final_score"] is None
    names_computed = {s["name"] for s in overall["submetrics"]["computed"]}
    names_omitted = {s["name"] for s in overall["submetrics"]["omitted"]}
    assert names_computed == {"accuracy", "language"}
    assert names_omitted == {"chatgpt", "match"}


def test_omitted_submetric_forces_final_score_none(val_records, base_predictions):
    """The acceptance-criterion-2 gate: chatgpt/match are always omitted (paid judge), so
    `final_score` must never be a partial sum — it's `null`, unconditionally, everywhere."""
    scored = _skip_if_scorer_unavailable(
        score_predictions, base_predictions, val_records, cfg=VQAEvalConfig()
    )
    assert scored["overall"]["final_score"] is None
    for category, result in scored["per_category"].items():
        assert result["final_score"] is None, category


def test_zero_item_subset_is_null_not_zero(val_records, base_predictions):
    """behavior is tag [0] only in the fixture — a category with zero *language*-tagged items
    must report language score None with a reason, never 0.0."""
    scored = _skip_if_scorer_unavailable(
        score_predictions, base_predictions, val_records, cfg=VQAEvalConfig()
    )
    behavior = scored["per_category"]["behavior"]
    language_entry = next(s for s in behavior["submetrics"]["computed"] if s["name"] == "language")
    assert language_entry["n_items"] == 0
    assert language_entry["score"] is None
    assert language_entry["reason"] == "0 items of this type in the split"


def test_annotate_per_item_flags_exact_match_only_for_tag_zero(base_predictions):
    rows = annotate_per_item(base_predictions)
    behavior_rows = [r for r in rows if r["category"] == "behavior"]
    assert all(r["exact_match"] is not None for r in behavior_rows)
    perception_rows = [r for r in rows if r["category"] == "perception"]
    assert all(r["exact_match"] is None for r in perception_rows)
    # the fixture's empty-prediction behavior row must be flagged incorrect, not silently true
    empty_row = next(r for r in behavior_rows if r["prediction"] == "")
    assert empty_row["exact_match"] is False


def test_build_and_write_vqa_metrics(tmp_path, val_records, base_predictions):
    gdp_cfg = load_config("configs/default.yaml")
    val_jsonl = tmp_path / "val.jsonl"
    val_jsonl.write_text("\n".join(r.qa_id for r in val_records))

    metrics = _skip_if_scorer_unavailable(
        build_vqa_metrics,
        model_role="base",
        predictions=base_predictions,
        records=val_records,
        cfg=VQAEvalConfig(),
        gdp_cfg=gdp_cfg,
        val_jsonl_path=val_jsonl,
        caveat=CAVEAT,
        split_provenance=SPLIT_PROVENANCE,
        is_synthetic=True,
    )
    assert metrics["model_role"] == "base"
    assert metrics["num_items"] == len(base_predictions)
    assert metrics["comparable_to_leaderboard"] is False
    assert metrics["caveat"] == CAVEAT
    assert metrics["generation"] == {"max_new_tokens": 128, "do_sample": False, "num_beams": 1}
    assert "scorer_sha256" in metrics
    assert "final_score_composition" in metrics
    assert metrics["config"]["seed"] == gdp_cfg.seed

    out_path = write_vqa_metrics(metrics, out_dir=tmp_path)
    assert out_path.is_file()
    assert json.loads(out_path.read_text())["model_role"] == "base"

    per_item_rows = annotate_per_item(base_predictions)
    per_item_path = write_per_item(per_item_rows, out_dir=tmp_path)
    lines = per_item_path.read_text().splitlines()
    assert len(lines) == len(base_predictions)
    assert json.loads(lines[0])["qa_id"] == base_predictions[0]["qa_id"]
