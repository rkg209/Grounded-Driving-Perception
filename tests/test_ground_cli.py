"""`gdp ground` CLI (spec 04 tasks 2-4): sample-frames, validate, freeze. No model is loaded by
any of these commands, so this file is intentionally NOT `model_heavy` (CLAUDE.md §4)."""

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
def clean_grounding_data():
    """`gdp ground sample-frames` writes under data/grounding_eval/, not runs/; clean up after."""
    data_dir = resolve("data/grounding_eval")
    existed_before = data_dir.is_dir()
    yield
    if data_dir.is_dir() and not existed_before:
        shutil.rmtree(data_dir)


def test_ground_sample_frames_writes_frames_json_and_overlays(clean_grounding_data):
    result = runner.invoke(
        app,
        [
            "ground",
            "sample-frames",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--n",
            "4",
        ],
    )
    assert result.exit_code == 0, result.stdout

    frames_path = resolve("data/grounding_eval/frames.json")
    assert frames_path.is_file()
    payload = json.loads(frames_path.read_text())
    assert payload["seed"] == 42
    assert len(payload["image_ids"]) == 4

    overlays_dir = resolve("data/grounding_eval/overlays")
    assert len(list(overlays_dir.iterdir())) == 4


def test_ground_sample_frames_same_seed_reproduces_same_frames(clean_grounding_data):
    first = runner.invoke(
        app,
        [
            "ground",
            "sample-frames",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--n",
            "3",
        ],
    )
    assert first.exit_code == 0, first.stdout
    first_ids = json.loads(resolve("data/grounding_eval/frames.json").read_text())["image_ids"]

    second = runner.invoke(
        app,
        [
            "ground",
            "sample-frames",
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
            "--n",
            "3",
        ],
    )
    assert second.exit_code == 0, second.stdout
    second_ids = json.loads(resolve("data/grounding_eval/frames.json").read_text())["image_ids"]

    assert first_ids == second_ids


def test_ground_validate_passes_on_the_fixture_phrase_set():
    result = runner.invoke(
        app,
        [
            "ground",
            "validate",
            "--phrases",
            FIXTURE_PHRASES,
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "8 phrases valid" in result.stdout


def test_ground_validate_fails_loudly_on_an_orphan_target(tmp_path):
    bad_phrases = tmp_path / "bad.json"
    bad_phrases.write_text(
        json.dumps(
            {
                "is_synthetic": True,
                "phrases": [
                    {
                        "image_id": 0,
                        "phrase": "the pedestrian",
                        "qualifier_type": "spatial",
                        "target_ann_id": 9999,
                        "author": "rahul",
                        "date": "2026-07-22",
                    }
                ],
            }
        )
    )
    result = runner.invoke(
        app,
        [
            "ground",
            "validate",
            "--phrases",
            str(bad_phrases),
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
        ],
    )
    assert result.exit_code == 1
    output = result.stdout + str(result.stderr or "")
    assert "does not exist" in output


def test_ground_freeze_then_tampering_makes_evaluate_time_assert_frozen_raise(tmp_path):
    phrases_path = tmp_path / "phrases_mini.json"
    phrases_path.write_text(resolve(FIXTURE_PHRASES).read_text())

    result = runner.invoke(
        app,
        [
            "ground",
            "freeze",
            "--phrases",
            str(phrases_path),
            "-c",
            "configs/default.yaml",
            "--dataset",
            "mini_bdd",
        ],
    )
    assert result.exit_code == 0, result.stdout

    lock_path = tmp_path / "phrases_mini.lock.json"
    assert lock_path.is_file()
    lock = json.loads(lock_path.read_text())
    assert lock["num_phrases"] == 8
    assert lock["author"] == "rahul"

    from gdp.ground.phrases import assert_frozen

    assert_frozen(phrases_path, lock_path)  # untouched: does not raise

    phrases_path.write_text(phrases_path.read_text().replace("curb", "curb2"))
    with pytest.raises(RuntimeError, match="has changed since it was frozen"):
        assert_frozen(phrases_path, lock_path)
