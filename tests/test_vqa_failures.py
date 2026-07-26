"""Spec 08 task 8: `gdp.vqa.failures` — deterministic stratified sampling + the markdown scaffold.

No model dependency. Synthetic per-item rows exercise sampling/determinism/stratification at
realistic scale (the real fixture only has 8 val items); the real `tests/fixtures/vqa_predictions`
+ `tests/fixtures/mini_drivelm` pair exercises `write_failures_md`'s image-copy + markdown shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from gdp.config import DRIVELM_CATEGORIES, load_config
from gdp.data.drivelm import convert_drivelm
from gdp.data.splits import load_official_splits, load_scene_meta
from gdp.vqa.failures import sample_failures, write_failures_md


def _synthetic_per_item(n_per_category: int = 20) -> list[dict]:
    rows = []
    for category in DRIVELM_CATEGORIES:
        for i in range(n_per_category):
            # every third item is "correct" (base==finetuned==GT), the rest are errors
            is_error = i % 3 != 0
            rows.append(
                {
                    "qa_id": f"{category}-{i:03d}",
                    "category": category,
                    "question": f"Question {i}?",
                    "gt_answer": "GT answer.",
                    "base_prediction": "wrong answer" if is_error else "GT answer.",
                    "finetuned_prediction": "wrong answer" if is_error else "GT answer.",
                    "image_path": "samples/CAM_FRONT/dummy.jpg",
                }
            )
    return rows


def test_sample_failures_is_deterministic_for_same_seed():
    per_item = _synthetic_per_item()
    a = sample_failures(per_item, seed=42, n=40, categories=DRIVELM_CATEGORIES)
    b = sample_failures(per_item, seed=42, n=40, categories=DRIVELM_CATEGORIES)
    assert [r["qa_id"] for r in a] == [r["qa_id"] for r in b]


def test_sample_failures_different_seed_can_differ():
    per_item = _synthetic_per_item()
    a = sample_failures(per_item, seed=1, n=40, categories=DRIVELM_CATEGORIES)
    b = sample_failures(per_item, seed=2, n=40, categories=DRIVELM_CATEGORIES)
    assert [r["qa_id"] for r in a] != [r["qa_id"] for r in b]


def test_sample_failures_stratified_no_category_vanishes():
    per_item = _synthetic_per_item()
    sample = sample_failures(per_item, seed=42, n=40, categories=DRIVELM_CATEGORIES)
    represented = {row["category"] for row in sample}
    assert represented == set(DRIVELM_CATEGORIES)


def test_sample_failures_only_returns_errors():
    per_item = _synthetic_per_item()
    sample = sample_failures(per_item, seed=42, n=40, categories=DRIVELM_CATEGORIES)
    assert all(row["finetuned_prediction"] != row["gt_answer"] for row in sample)


def test_sample_failures_respects_n():
    per_item = _synthetic_per_item()
    sample = sample_failures(per_item, seed=42, n=40, categories=DRIVELM_CATEGORIES)
    assert len(sample) == 40


def test_sample_failures_uses_error_on_field_when_present():
    per_item = _synthetic_per_item(n_per_category=5)
    for row in per_item:
        row["finetuned_exact_match"] = row["finetuned_prediction"] == row["gt_answer"]
    sample = sample_failures(
        per_item, seed=1, n=8, categories=DRIVELM_CATEGORIES, error_on="finetuned_exact_match"
    )
    assert all(row["finetuned_exact_match"] is False for row in sample)


def _merged_real_per_item(val_records) -> list[dict]:
    base_rows = [
        json.loads(line)
        for line in Path("tests/fixtures/vqa_predictions/base_mini.jsonl").read_text().splitlines()
    ]
    finetuned_rows = {
        json.loads(line)["qa_id"]: json.loads(line)
        for line in Path("tests/fixtures/vqa_predictions/finetuned_mini.jsonl")
        .read_text()
        .splitlines()
    }
    image_by_qa_id = {r.qa_id: r.image_paths["CAM_FRONT"] for r in val_records}
    merged = []
    for row in base_rows:
        ft = finetuned_rows[row["qa_id"]]
        merged.append(
            {
                "qa_id": row["qa_id"],
                "category": row["category"],
                "question": row["question"],
                "gt_answer": row["gt_answer"],
                "base_prediction": row["prediction"],
                "finetuned_prediction": ft["prediction"],
                "image_path": image_by_qa_id[row["qa_id"]],
            }
        )
    return merged


def test_write_failures_md_end_to_end(tmp_path):
    cfg = load_config("configs/default.yaml").drivelm
    raw = json.loads(cfg.annotations_path().read_text())
    scene_meta = load_scene_meta(cfg.scene_meta_path())
    splits = load_official_splits()
    _, val_records, _ = convert_drivelm(
        raw,
        scene_meta=scene_meta,
        splits=splits,
        images_root=cfg.nuscenes_root_path(),
        categories=tuple(cfg.categories),
        is_synthetic=True,
    )
    merged = _merged_real_per_item(val_records)
    # the fixture's finetuned predictions are all clean by design (task 3 shows the fix, not a
    # failure) — inject one realistic fine-tuned miss so this test exercises the write path on an
    # actual sampled item, the way a real (imperfect) evaluation run would.
    merged[0] = {**merged[0], "finetuned_prediction": "a completely wrong answer"}
    sample = sample_failures(merged, seed=42, n=4, categories=tuple(DRIVELM_CATEGORIES))
    assert len(sample) > 0

    md_path = write_failures_md(sample, out_dir=tmp_path, images_root=cfg.nuscenes_root_path())
    text = md_path.read_text()
    assert "Generated fields" in text
    assert "Hand-written fields" in text
    assert "**Category:** " in text
    assert "**Comment:** " in text
    for row in sample:
        assert row["qa_id"] in text

    copied = list((tmp_path / "failures").glob("*.jpg"))
    assert len(copied) == len(sample)
