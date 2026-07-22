"""`gdp ground compare` (spec 04 task 7). Pure JSON-in, JSON-out — no model, not `model_heavy`."""

from __future__ import annotations

import json
import shutil

import pytest
from typer.testing import CliRunner

from gdp.cli import app
from gdp.paths import resolve

runner = CliRunner()

BASE_METRICS = {
    "dataset": "mini_bdd",
    "is_synthetic": True,
    "is_self_built_benchmark": True,
    "caveat": "self-built",
    "model_id": "IDEA-Research/grounding-dino-tiny",
    "num_phrases": 8,
    "phrases_sha256": "db124fd117abc47cbf685b75e7f2bcaedff604aac425eb56f2da849a0fd54376",
    "frozen_at": "2026-07-22T06:33:52.932425+00:00",
    "box_threshold": 0.25,
    "chosen_on": "train",
    "iou_threshold": 0.5,
    "per_type": {
        "spatial": {"n": 2, "correct": 0, "accuracy": 0.0},
        "attribute": {"n": 2, "correct": 0, "accuracy": 0.0},
        "relational": {"n": 2, "correct": 0, "accuracy": 0.0},
        "negative": {"n": 2, "correct": 0, "accuracy": 0.0},
    },
    "positives": {"n": 6, "correct": 0, "accuracy": 0.0},
    "negatives": {"n": 2, "correct": 0, "accuracy": 0.0},
    "overall": {"n": 8, "correct": 0, "accuracy": 0.0},
}


@pytest.fixture
def clean_04_grounding_runs():
    runs_dir = resolve("runs/04-grounding")
    before = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()
    yield
    after = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()
    for new_dir in after - before:
        shutil.rmtree(new_dir)


def test_ground_compare_writes_grounding_comparison_json(clean_04_grounding_runs, tmp_path):
    zeroshot_path = tmp_path / "zeroshot_metrics.json"
    finetuned_path = tmp_path / "finetuned_metrics.json"
    zeroshot_path.write_text(json.dumps(BASE_METRICS))
    finetuned_path.write_text(
        json.dumps(
            {
                **BASE_METRICS,
                "model_id": "runs/03-finetune/20260101/checkpoint-500",
                "overall": {"n": 8, "correct": 3, "accuracy": 0.375},
            }
        )
    )

    result = runner.invoke(
        app,
        ["ground", "compare", "--zeroshot", str(zeroshot_path), "--finetuned", str(finetuned_path)],
    )
    assert result.exit_code == 0, result.stdout

    runs_dir = resolve("runs/04-grounding")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    comparison = json.loads((latest / "grounding_comparison.json").read_text())
    assert comparison["accuracy_zeroshot"] == 0.0
    assert comparison["accuracy_finetuned"] == pytest.approx(0.375)
    assert comparison["accuracy_delta"] == pytest.approx(0.375)
    assert comparison["is_self_built_benchmark"] is True


def test_ground_compare_refuses_mismatched_phrase_sets(clean_04_grounding_runs, tmp_path):
    zeroshot_path = tmp_path / "zeroshot_metrics.json"
    finetuned_path = tmp_path / "finetuned_metrics.json"
    zeroshot_path.write_text(json.dumps(BASE_METRICS))
    finetuned_path.write_text(json.dumps({**BASE_METRICS, "phrases_sha256": "different"}))

    result = runner.invoke(
        app,
        ["ground", "compare", "--zeroshot", str(zeroshot_path), "--finetuned", str(finetuned_path)],
    )
    assert result.exit_code == 1
    output = result.stdout + str(result.stderr or "")
    assert "refusing to compare mismatched grounding runs" in output
