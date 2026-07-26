"""Spec 08 task 7: `gdp.vqa.compare` — the H8 mismatch gate and the `regressions` field.

No model dependency; builds the two `metrics.json`-shaped dicts directly via `gdp.vqa.score` on
the hand-built fixture predictions rather than exercising `gdp.vqa.official`'s real scorer (that
round-trip is already covered by `tests/test_vqa_score.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gdp.config import VQAEvalConfig, load_config
from gdp.data.drivelm import convert_drivelm
from gdp.data.drivelm_stats import CAVEAT, SPLIT_PROVENANCE
from gdp.data.splits import load_official_splits, load_scene_meta
from gdp.vqa.compare import build_vqa_comparison, write_vqa_comparison
from gdp.vqa.official import OfficialScorerUnavailable
from gdp.vqa.score import build_vqa_metrics


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


def _load_predictions(name: str) -> list[dict]:
    path = Path("tests/fixtures/vqa_predictions") / name
    return [json.loads(line) for line in path.read_text().splitlines()]


def _build_metrics(role, predictions, val_records, val_jsonl_path):
    gdp_cfg = load_config("configs/default.yaml")
    try:
        return build_vqa_metrics(
            model_role=role,
            predictions=predictions,
            records=val_records,
            cfg=VQAEvalConfig(),
            gdp_cfg=gdp_cfg,
            val_jsonl_path=val_jsonl_path,
            caveat=CAVEAT,
            split_provenance=SPLIT_PROVENANCE,
            is_synthetic=True,
        )
    except OfficialScorerUnavailable as exc:
        pytest.skip(f"vendored scorer deps not installed: {exc}")


@pytest.fixture(scope="module")
def val_jsonl(tmp_path_factory, val_records):
    path = tmp_path_factory.mktemp("compare") / "val.jsonl"
    path.write_text("\n".join(r.qa_id for r in val_records))
    return path


@pytest.fixture(scope="module")
def base_metrics(val_records, val_jsonl):
    return _build_metrics("base", _load_predictions("base_mini.jsonl"), val_records, val_jsonl)


@pytest.fixture(scope="module")
def finetuned_metrics(val_records, val_jsonl):
    return _build_metrics(
        "finetuned", _load_predictions("finetuned_mini.jsonl"), val_records, val_jsonl
    )


def test_build_vqa_comparison_matches_same_run(base_metrics, finetuned_metrics):
    comparison = build_vqa_comparison(base_metrics, finetuned_metrics)
    assert comparison["final_score"] is None
    assert set(comparison["per_category_accuracy"]) == {
        "perception",
        "prediction",
        "planning",
        "behavior",
    }
    # the finetuned fixture's empty-prediction defect (task 3) is fixed -> behavior improves
    assert "behavior" not in comparison["regressions"]
    assert comparison["overall_accuracy"]["after"] >= comparison["overall_accuracy"]["before"]


def test_build_vqa_comparison_raises_on_differing_generation(base_metrics, finetuned_metrics):
    mutated = dict(finetuned_metrics)
    mutated["generation"] = {**finetuned_metrics["generation"], "max_new_tokens": 64}
    with pytest.raises(ValueError, match="generation"):
        build_vqa_comparison(base_metrics, mutated)


def test_build_vqa_comparison_raises_on_differing_scorer_sha(base_metrics, finetuned_metrics):
    mutated = dict(finetuned_metrics)
    mutated["scorer_sha256"] = "deadbeef"
    with pytest.raises(ValueError, match="scorer_sha256"):
        build_vqa_comparison(base_metrics, mutated)


def test_build_vqa_comparison_raises_on_differing_val_hash(base_metrics, finetuned_metrics):
    mutated = dict(finetuned_metrics)
    mutated["val_jsonl_sha256"] = "deadbeef"
    with pytest.raises(ValueError, match="val_jsonl_sha256"):
        build_vqa_comparison(base_metrics, mutated)


def test_build_vqa_comparison_raises_on_differing_num_items(base_metrics, finetuned_metrics):
    mutated = dict(finetuned_metrics)
    mutated["num_items"] = finetuned_metrics["num_items"] - 1
    with pytest.raises(ValueError, match="num_items"):
        build_vqa_comparison(base_metrics, mutated)


def test_write_vqa_comparison(tmp_path, base_metrics, finetuned_metrics):
    comparison = build_vqa_comparison(base_metrics, finetuned_metrics)
    path = write_vqa_comparison(comparison, out_dir=tmp_path)
    assert path.is_file()
    assert json.loads(path.read_text())["regressions"] == comparison["regressions"]
