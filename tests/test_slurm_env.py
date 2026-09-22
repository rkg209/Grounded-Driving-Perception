"""Guards on the cluster contract every SLURM job script depends on.

None of this tests a metric. It tests the four PARAM Rudra facts that killed (or would have
killed) a job in progress_report.md [SEQ-0122]: model weights must not be written into a full
$HOME, compute nodes are offline, Triton needs a sysroot, and `uv` is not on a job's PATH. A
regression here costs a queue slot and an hour, silently, so it is cheaper to assert it.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from gdp.paths import _git_sha_from_files, git_sha, resolve

SLURM_DIR = resolve("scripts/slurm")
JOB_SCRIPTS = sorted(SLURM_DIR.glob("*.slurm"))


def test_job_scripts_exist():
    """Five specs have cluster steps; if one disappears, the rest of this file silently passes."""
    assert {p.name for p in JOB_SCRIPTS} == {
        "zeroshot_eval.slurm",
        "finetune_detector.slurm",
        "grounding_eval.slurm",
        "finetune_vlm.slurm",
        "vqa_eval.slurm",
    }


@pytest.mark.parametrize("script", JOB_SCRIPTS, ids=lambda p: p.name)
def test_job_script_sources_cluster_env(script):
    assert "source scripts/slurm/cluster_env.sh" in script.read_text()


@pytest.mark.parametrize("script", JOB_SCRIPTS, ids=lambda p: p.name)
def test_job_script_does_not_point_hf_home_at_home(script):
    """`${SCRATCH:-$HOME}/hf` resolved to $HOME on Rudra, whose group quota is full — the job died
    mid-download. cluster_env.sh owns HF_HOME now; no script may set its own."""
    assert "SCRATCH:-$HOME" not in script.read_text()


@pytest.mark.parametrize("script", JOB_SCRIPTS, ids=lambda p: p.name)
def test_job_script_documents_its_log_directory(script):
    """SLURM will not create the --output directory, and fails the job without a useful message."""
    text = script.read_text()
    assert "mkdir -p runs/" in text


def test_cluster_env_covers_every_rudra_blocker():
    text = (SLURM_DIR / "cluster_env.sh").read_text()
    for needed in (
        "HF_HOME",  # weights off the full $HOME quota
        "HF_HUB_OFFLINE",  # compute nodes have no DNS
        "TRANSFORMERS_OFFLINE",
        "C_INCLUDE_PATH",  # Triton compiles on the first CUDA forward pass
        "UV_NO_SYNC",  # a re-sync silently undoes the cu126 torch swap
        "TMPDIR",
    ):
        assert needed in text, needed


def test_cluster_env_is_sourced_not_executed():
    """It configures the *calling* shell; run with a shebang, its exports die with the subshell."""
    assert "Sourced, never executed" in (SLURM_DIR / "cluster_env.sh").read_text()


def test_git_sha_resolves_without_the_git_binary(monkeypatch):
    """Rudra's compute nodes have no `git`, which stamped every cluster artifact "unknown" and cost
    it its provenance (H8). The fallback reads .git directly."""

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git: command not found")

    monkeypatch.setattr(subprocess, "run", no_git)
    sha = git_sha()
    assert sha != "unknown"
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_git_sha_file_fallback_matches_the_git_binary():
    """The fallback must return the same commit git does, not merely *a* plausible sha."""
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=resolve("."),
    ).stdout.strip()
    assert _git_sha_from_files() == expected


def test_git_sha_never_raises_outside_a_repo(monkeypatch, tmp_path):
    """A stamped artifact must never be *blocked* by provenance lookup failing."""
    monkeypatch.chdir(tmp_path)
    assert isinstance(git_sha(), str)


@pytest.mark.skipif(os.name != "posix", reason="shell syntax check needs a POSIX shell")
def test_cluster_env_is_valid_shell():
    """A syntax error here would fail all five jobs at the first line, after queueing."""
    subprocess.run(["bash", "-n", str(SLURM_DIR / "cluster_env.sh")], check=True)


@pytest.mark.skipif(os.name != "posix", reason="shell syntax check needs a POSIX shell")
@pytest.mark.parametrize("script", JOB_SCRIPTS, ids=lambda p: p.name)
def test_job_script_is_valid_shell(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def _source_cluster_env(tmp_path, env_overrides):
    """Source cluster_env.sh in a clean shell and read back what it exported."""
    scr = tmp_path / "scr"
    home = tmp_path / "home"
    scr.mkdir()
    home.mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "USER": os.environ.get("USER", "tester"),
        "HOME": str(home),
        "SCR": str(scr),
        **env_overrides,
    }
    out = subprocess.run(
        ["bash", "-c", "source scripts/slurm/cluster_env.sh > /dev/null && env"],
        cwd=resolve("."),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line), str(scr), str(home)


def test_hf_home_is_redirected_away_from_a_stale_home_export(tmp_path):
    """The original bug in another shape: HF_HOME inherited from ~/.bashrc still pointing at the
    full home quota. Deferring to an inherited value would re-create the failure."""
    env, scr, home = _source_cluster_env(tmp_path, {"HF_HOME": str(tmp_path / "home" / "hf")})
    assert env["HF_HOME"] == f"{scr}/.cache/huggingface"
    assert not env["HF_HOME"].startswith(home)


def test_hf_home_respects_a_deliberate_scratch_override(tmp_path):
    """A login-node pre-download and the job must be able to share one cache."""
    shared = str(tmp_path / "scr" / "shared-hf")
    env, _, _ = _source_cluster_env(tmp_path, {"HF_HOME": shared})
    assert env["HF_HOME"] == shared


def test_tmpdir_is_redirected_off_node_local_tmp(tmp_path):
    """SLURM commonly sets TMPDIR=/tmp: small, node-local, and wiped at job end."""
    env, scr, _ = _source_cluster_env(tmp_path, {"TMPDIR": "/tmp"})
    assert env["TMPDIR"] == f"{scr}/tmp"


def test_offline_flags_can_be_opted_out_of(tmp_path):
    """Off Rudra, a cluster whose compute nodes have internet must not be forced offline."""
    env, _, _ = _source_cluster_env(tmp_path, {"GDP_ALLOW_NETWORK": "1"})
    assert "HF_HUB_OFFLINE" not in env
