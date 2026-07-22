"""`gdp ground evaluate` (spec 04 task 6). Only the happy-path test below actually constructs a
`GroundingDinoDetector` (one real model load) — CLAUDE.md §4's memory-hygiene rule requires
`model_heavy` on that one specifically, not the whole file: the two refusal tests exit via
`assert_frozen`'s hash check *before* the detector is ever built, so they stay in the default,
fast `uv run pytest` run.
"""

from __future__ import annotations

import json
import shutil

import pytest
from typer.testing import CliRunner

from gdp.cli import app
from gdp.paths import resolve

runner = CliRunner()

FIXTURE_PHRASES = "tests/fixtures/grounding/phrases_mini.json"


@pytest.fixture
def clean_04_grounding_runs():
    runs_dir = resolve("runs/04-grounding")
    before = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()
    yield
    after = set(runs_dir.iterdir()) if runs_dir.is_dir() else set()
    for new_dir in after - before:
        shutil.rmtree(new_dir)


@pytest.mark.model_heavy
def test_ground_evaluate_writes_metrics_and_per_phrase_with_full_provenance(
    clean_04_grounding_runs,
):
    result = runner.invoke(
        app,
        [
            "ground",
            "evaluate",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--phrases",
            FIXTURE_PHRASES,
            "--box-threshold",
            "0.25",
            "--chosen-on",
            "train",
        ],
    )
    assert result.exit_code == 0, result.stdout

    runs_dir = resolve("runs/04-grounding")
    latest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    metrics = json.loads((latest / "metrics.json").read_text())
    per_phrase = json.loads((latest / "per_phrase.json").read_text())

    assert metrics["is_synthetic"] is True
    assert metrics["is_self_built_benchmark"] is True
    assert "caveat" in metrics
    assert metrics["num_phrases"] == 8
    assert (
        metrics["phrases_sha256"]
        == "db124fd117abc47cbf685b75e7f2bcaedff604aac425eb56f2da849a0fd54376"
    )
    assert metrics["chosen_on"] == "train"
    assert metrics["box_threshold"] == 0.25
    assert set(metrics["per_type"]) == {"spatial", "attribute", "relational", "negative"}
    for key in ("per_type", "positives", "negatives", "overall"):
        assert "n" in metrics[key] if key != "per_type" else True
    assert metrics["positives"]["n"] == 6
    assert metrics["negatives"]["n"] == 2
    assert metrics["overall"]["n"] == 8

    assert len(per_phrase) == 8
    for record in per_phrase:
        assert {"phrase", "qualifier_type", "image_id", "correct"} <= record.keys()


def test_ground_evaluate_refuses_a_tampered_phrases_file(clean_04_grounding_runs, tmp_path):
    tampered = tmp_path / "phrases_mini.json"
    payload = json.loads(resolve(FIXTURE_PHRASES).read_text())
    payload["phrases"][0]["phrase"] = "the pedestrian near the curb, tampered"
    tampered.write_text(json.dumps(payload))
    # Carry over the *original* fixture's lock file unchanged, so the hash mismatch is real.
    shutil.copy(
        resolve(FIXTURE_PHRASES).with_name("phrases_mini.lock.json"),
        tmp_path / "phrases_mini.lock.json",
    )

    result = runner.invoke(
        app,
        [
            "ground",
            "evaluate",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--phrases",
            str(tampered),
            "--box-threshold",
            "0.25",
            "--chosen-on",
            "train",
        ],
    )
    assert result.exit_code == 1
    output = result.stdout + str(result.stderr or "")
    assert "has changed since it was frozen" in output


def test_ground_evaluate_missing_lock_file_refuses(clean_04_grounding_runs, tmp_path):
    unfrozen = tmp_path / "phrases_mini.json"
    unfrozen.write_text(resolve(FIXTURE_PHRASES).read_text())
    result = runner.invoke(
        app,
        [
            "ground",
            "evaluate",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--phrases",
            str(unfrozen),
        ],
    )
    assert result.exit_code == 1
    assert "no lock file" in (result.stdout + str(result.stderr or ""))
