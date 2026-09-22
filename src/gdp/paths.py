"""Repo-relative path resolution.

Everything in this project is addressed relative to the repo root so that the same config
works on the M4 laptop and on the SLURM cluster without edits.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path


def git_sha() -> str:
    """The current commit, for stamping into every metrics.json (CLAUDE.md §8).

    Falls back to reading `.git` directly because PARAM Rudra's compute nodes have no `git`
    binary: every artifact produced by an `srun`/`sbatch` step there was stamped
    `"git_sha": "unknown"` (see progress_report.md [SEQ-0122]), which silently costs every
    cluster-side number its provenance — exactly the number H8 needs to be traceable.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return _git_sha_from_files()


def _git_sha_from_files() -> str:
    """Resolve HEAD by reading `.git/` — no `git` binary, no subprocess.

    Handles the two shapes `.git/HEAD` takes: a symbolic ref into `refs/heads/...` (the normal
    case, resolved via the ref file or `packed-refs`) and a bare detached-HEAD sha.
    """
    try:
        git_dir = repo_root() / ".git"
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head if head else "unknown"
        ref = head[len("ref: ") :].strip()
        ref_file = git_dir / ref
        if ref_file.is_file():
            return ref_file.read_text().strip()
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line.endswith(f" {ref}"):
                    return line.split(" ", 1)[0]
        return "unknown"
    except (OSError, RuntimeError):
        return "unknown"


def repo_root() -> Path:
    """Walk up from this file until the directory holding pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate repo root (no pyproject.toml found in any parent).")


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the repo root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (repo_root() / p)


def run_dir(spec: str, *, create: bool = True) -> Path:
    """A timestamped output directory for one spec's run: runs/<spec>/<timestamp>/.

    Metrics always land in a file here (CLAUDE.md §8) so a result can never exist only in prose.
    """
    d = resolve("runs") / spec / datetime.now().strftime("%Y%m%d-%H%M%S")
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d
