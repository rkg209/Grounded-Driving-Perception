"""Building `docs/metrics_snapshot.json` — the committed intermediate between `runs/` and README.

Three statuses, and the third one is the point (spec 10 decision 2):

- `present`   — a real artifact from a real run; its numbers may be rendered.
- `pending`   — no artifact yet; the table renders a named blocker, never a blank.
- `synthetic` — an artifact exists but was computed on `tests/fixtures/` (`is_synthetic: true`).
  It renders *exactly like* `pending`. Printing a fixture number as a finding is the H7 failure
  the register exists to stop, so no flag, option, or override turns it into a result.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from gdp.paths import git_sha, resolve
from gdp.report.sources import STAGES, StageSource, latest_run_dir, repo_relative, sha256_file

SNAPSHOT_VERSION = 1
SNAPSHOT_PATH = "docs/metrics_snapshot.json"

SYNTHETIC_REASON = "last run was on tests/fixtures/ — not a result (H7)"

Status = Literal["present", "pending", "synthetic"]


def _contains_synthetic(node: Any) -> bool:
    """`is_synthetic: true` anywhere in the copied subset condemns the whole stage.

    Recursive because spec 08's `metrics.json` carries the flag one level down, per model
    (`models.base.is_synthetic`), while spec 05's carries it at the top.
    """
    if isinstance(node, dict):
        if node.get("is_synthetic") is True:
            return True
        return any(_contains_synthetic(v) for v in node.values())
    if isinstance(node, list):
        return any(_contains_synthetic(v) for v in node)
    return False


def _pending(source: StageSource, reason: str) -> dict[str, Any]:
    return {
        "status": "pending",
        "title": source.title,
        "spec": source.spec,
        "blocker": source.blocker,
        "reason": reason,
        "source": None,
        "created": None,
        "sources": {},
        "data": {},
    }


def build_stage(name: str, source: StageSource, *, runs_root: Path | None = None) -> dict[str, Any]:
    """One stage entry. `name` is unused beyond error messages — kept for symmetry with `STAGES`."""

    run_dir = latest_run_dir(source.spec, source.primary.filename, runs_root=runs_root)
    if run_dir is None:
        return _pending(
            source,
            f"no `runs/{source.spec}/*/{source.primary.filename}` exists yet — "
            f"blocked on: {source.blocker}",
        )

    data: dict[str, Any] = {}
    sources: dict[str, dict[str, str]] = {}
    for spec_file in source.files:
        path = run_dir / spec_file.filename
        if not path.is_file():
            return _pending(
                source,
                f"`{repo_relative(run_dir)}` has no `{spec_file.filename}` — the run is "
                f"incomplete; blocked on: {source.blocker}",
            )
        raw = json.loads(path.read_text())
        data[spec_file.key] = spec_file.subset(raw)
        sources[spec_file.key] = {
            "path": repo_relative(path),
            "sha256": sha256_file(path),
        }

    primary_path = run_dir / source.primary.filename
    created = _stage_created(data)
    status: Status = "synthetic" if _contains_synthetic(data) else "present"
    return {
        "status": status,
        "title": source.title,
        "spec": source.spec,
        "blocker": source.blocker,
        "reason": SYNTHETIC_REASON if status == "synthetic" else None,
        "source": repo_relative(primary_path),
        "sha256": sha256_file(primary_path),
        "created": created,
        "sources": sources,
        # `data` is a verbatim subset of the artifacts — never recomputed, never rounded here.
        "data": data,
    }


def _stage_created(data: dict[str, Any]) -> str | None:
    """The *source's* own created stamp, not the snapshot's. A table footline dated with the
    snapshot's build time would say the run happened when the README was regenerated."""
    for node in data.values():
        if isinstance(node, dict):
            for key in ("created", "finetuned_created", "base_created"):
                if node.get(key):
                    return str(node[key])
    return None


def build_snapshot(*, runs_root: Path | None = None) -> dict[str, Any]:
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "created": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "stages": {
            name: build_stage(name, source, runs_root=runs_root) for name, source in STAGES.items()
        },
    }


def write_snapshot(snapshot: dict[str, Any], *, path: str | Path | None = None) -> Path:
    out = Path(path) if path is not None else resolve(SNAPSHOT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2) + "\n")
    return out


def load_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    src = Path(path) if path is not None else resolve(SNAPSHOT_PATH)
    if not src.is_file():
        raise FileNotFoundError(
            f"{repo_relative(src)} does not exist — run `uv run gdp report snapshot` first"
        )
    snapshot = json.loads(src.read_text())
    version = snapshot.get("snapshot_version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(
            f"{repo_relative(src)} is snapshot_version {version!r}, this build expects "
            f"{SNAPSHOT_VERSION} — re-run `uv run gdp report snapshot`"
        )
    return snapshot
