"""Spec 10 tasks 1–2: the run-file resolver and the three snapshot statuses.

No model is loaded anywhere in this file, so it is intentionally NOT `model_heavy` (CLAUDE.md §4).
"""

from __future__ import annotations

import json

import pytest

from gdp.paths import repo_root
from gdp.report import snapshot as snap
from gdp.report.sources import STAGES, latest_run_dir, repo_relative, sha256_file

FIXTURE_RUNS = repo_root() / "tests" / "fixtures" / "report" / "runs"


# --------------------------------------------------------------------------- sources


def test_sources_latest_run_dir_finds_the_fixture_run():
    found = latest_run_dir("03-finetune", "comparison.json", runs_root=FIXTURE_RUNS)
    assert found == FIXTURE_RUNS / "03-finetune" / "20260801-101500"


def test_sources_latest_run_dir_is_none_for_an_unknown_spec():
    assert latest_run_dir("99-nope", "comparison.json", runs_root=FIXTURE_RUNS) is None


def test_sources_latest_run_dir_skips_dirs_without_the_file(tmp_path):
    """A run directory that exists but never produced the artifact is not a candidate — otherwise
    a crashed run would shadow the last good one."""
    (tmp_path / "03-finetune" / "20260101-000000").mkdir(parents=True)
    good = tmp_path / "03-finetune" / "20250101-000000"
    good.mkdir(parents=True)
    (good / "comparison.json").write_text("{}")
    assert latest_run_dir("03-finetune", "comparison.json", runs_root=tmp_path) == good


def test_sources_latest_run_dir_picks_the_newest_file_mtime(tmp_path):
    older = tmp_path / "03-finetune" / "20260101-000000"
    newer = tmp_path / "03-finetune" / "20250101-000000"
    for d in (older, newer):
        d.mkdir(parents=True)
        (d / "comparison.json").write_text("{}")
    import os

    os.utime(older / "comparison.json", (1000, 1000))
    os.utime(newer / "comparison.json", (2000, 2000))
    assert latest_run_dir("03-finetune", "comparison.json", runs_root=tmp_path) == newer


def test_sources_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "x.json"
    path.write_bytes(b'{"a": 1}')
    assert sha256_file(path) == hashlib.sha256(b'{"a": 1}').hexdigest()


def test_sources_stage_04_uses_its_own_comparison_filename():
    """Spec 04 writes `grounding_comparison.json`, spec 03 writes `comparison.json` — the one
    filename difference that a guessed convention would get silently wrong."""
    assert STAGES["grounding"].primary.filename == "grounding_comparison.json"
    assert STAGES["stage1_detection"].primary.filename == "comparison.json"


def test_sources_every_stage_names_a_blocker():
    for name, source in STAGES.items():
        assert source.blocker.strip(), f"{name} has no named blocker"


def test_repo_relative_is_posix_and_repo_rooted():
    rel = repo_relative(repo_root() / "docs" / "metrics_snapshot.json")
    assert rel == "docs/metrics_snapshot.json"


# --------------------------------------------------------------------------- snapshot


@pytest.fixture(scope="module")
def fixture_snapshot():
    return snap.build_snapshot(runs_root=FIXTURE_RUNS)


def test_snapshot_has_all_four_stages(fixture_snapshot):
    assert set(fixture_snapshot["stages"]) == set(STAGES)


def test_snapshot_fixture_stages_are_present(fixture_snapshot):
    for name, entry in fixture_snapshot["stages"].items():
        assert entry["status"] == "present", f"{name}: {entry['reason']}"


def test_snapshot_records_source_path_and_sha(fixture_snapshot):
    entry = fixture_snapshot["stages"]["stage1_detection"]
    assert entry["source"].endswith("20260801-101500/comparison.json")
    assert len(entry["sha256"]) == 64
    assert entry["sources"]["comparison"]["sha256"] == entry["sha256"]


def test_snapshot_copies_only_the_declared_fields(fixture_snapshot):
    data = fixture_snapshot["stages"]["deployment"]["data"]["metrics"]
    assert "config" not in data, "the whole Config blob must not enter the snapshot"
    assert data["variants"]["int8-onnx"]["map"] == 0.3301


def test_snapshot_copies_values_verbatim(fixture_snapshot):
    raw = json.loads(
        (FIXTURE_RUNS / "03-finetune" / "20260801-101500" / "comparison.json").read_text()
    )
    data = fixture_snapshot["stages"]["stage1_detection"]["data"]["comparison"]
    for key in ("map_zeroshot", "map_finetuned", "map_delta", "per_class_ap", "regressions"):
        assert data[key] == raw[key]


def test_snapshot_created_is_the_sources_own_stamp_not_the_builds(fixture_snapshot):
    entry = fixture_snapshot["stages"]["stage1_detection"]
    assert entry["created"] == "2026-08-01T10:15:00+00:00"


def test_snapshot_normalises_the_two_vqa_metrics_shapes(fixture_snapshot):
    models = fixture_snapshot["stages"]["stage2_vqa"]["data"]["metrics"]["models"]
    assert set(models) == {"base", "finetuned"}
    assert models["finetuned"]["comparable_to_leaderboard"] is False


def test_snapshot_vqa_flat_shape_is_normalised_too(tmp_path):
    run = tmp_path / "08-vqa" / "20260101-000000"
    run.mkdir(parents=True)
    (run / "comparison.json").write_text("{}")
    (run / "metrics.json").write_text(
        json.dumps({"model_role": "finetuned", "num_items": 3, "is_synthetic": False})
    )
    entry = snap.build_stage("stage2_vqa", STAGES["stage2_vqa"], runs_root=tmp_path)
    assert set(entry["data"]["metrics"]["models"]) == {"finetuned"}


def test_snapshot_missing_artifact_is_pending_with_its_blocker(tmp_path):
    entry = snap.build_stage("stage1_detection", STAGES["stage1_detection"], runs_root=tmp_path)
    assert entry["status"] == "pending"
    assert entry["data"] == {}
    assert STAGES["stage1_detection"].blocker in entry["reason"]


def test_snapshot_incomplete_run_is_pending(tmp_path):
    """`metrics.json` without its `latency.json` is half a deployment result — H8 says half is
    none, so the whole stage stays pending."""
    run = tmp_path / "05-deploy" / "20260101-000000"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"is_synthetic": False}))
    entry = snap.build_stage("deployment", STAGES["deployment"], runs_root=tmp_path)
    assert entry["status"] == "pending"
    assert "latency.json" in entry["reason"]


def test_snapshot_is_synthetic_becomes_the_synthetic_status(tmp_path):
    run = tmp_path / "05-deploy" / "20260101-000000"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"is_synthetic": True, "variants": {}}))
    (run / "latency.json").write_text(json.dumps({"hardware": {}, "variants": {}}))
    entry = snap.build_stage("deployment", STAGES["deployment"], runs_root=tmp_path)
    assert entry["status"] == "synthetic"
    assert entry["reason"] == snap.SYNTHETIC_REASON


def test_snapshot_is_synthetic_is_detected_when_nested(tmp_path):
    """Spec 08 carries the flag per model, one level down — a top-level-only check would publish
    a fixture VQA number as a result."""
    run = tmp_path / "08-vqa" / "20260101-000000"
    run.mkdir(parents=True)
    (run / "comparison.json").write_text(json.dumps({"num_items": 8}))
    (run / "metrics.json").write_text(
        json.dumps({"models": {"base": {"is_synthetic": True}, "finetuned": {}}})
    )
    entry = snap.build_stage("stage2_vqa", STAGES["stage2_vqa"], runs_root=tmp_path)
    assert entry["status"] == "synthetic"


def test_snapshot_write_then_load_round_trips(tmp_path, fixture_snapshot):
    path = snap.write_snapshot(fixture_snapshot, path=tmp_path / "metrics_snapshot.json")
    assert snap.load_snapshot(path) == fixture_snapshot


def test_snapshot_load_rejects_a_foreign_version(tmp_path):
    path = tmp_path / "metrics_snapshot.json"
    path.write_text(json.dumps({"snapshot_version": 999, "stages": {}}))
    with pytest.raises(ValueError, match="snapshot_version"):
        snap.load_snapshot(path)


def test_snapshot_load_reports_a_missing_file_with_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="gdp report snapshot"):
        snap.load_snapshot(tmp_path / "nope.json")


def test_committed_snapshot_is_loadable_and_covers_every_stage():
    """The committed `docs/metrics_snapshot.json` is what a clean clone renders from."""
    committed = snap.load_snapshot()
    assert set(committed["stages"]) == set(STAGES)
