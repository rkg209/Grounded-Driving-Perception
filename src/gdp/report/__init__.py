"""Spec 10 — the write-up generator.

The README's metrics tables are *rendered*, never typed. The pipeline is deliberately two-step:

    runs/<spec>/<ts>/*.json  --(gdp report snapshot)-->  docs/metrics_snapshot.json
    docs/metrics_snapshot.json  --(gdp report render)-->  README.md generated blocks

`runs/` is gitignored, so a clean clone cannot see a single `metrics.json`; the committed snapshot
is what makes "every number traces to a file the code wrote" checkable by a test in that clone.
The renderer therefore never touches `runs/` — if a number is in the README, it is in the snapshot,
and the snapshot records the source path and its sha256 so drift is detectable.
"""

from __future__ import annotations

from gdp.report.inject import check_blocks, extract_blocks, inject_blocks, marker_spans
from gdp.report.snapshot import (
    SNAPSHOT_PATH,
    SNAPSHOT_VERSION,
    build_snapshot,
    load_snapshot,
    write_snapshot,
)
from gdp.report.sources import STAGES, StageSource, latest_run_dir, sha256_file
from gdp.report.tables import render_all

__all__ = [
    "SNAPSHOT_PATH",
    "SNAPSHOT_VERSION",
    "STAGES",
    "StageSource",
    "build_snapshot",
    "check_blocks",
    "extract_blocks",
    "inject_blocks",
    "latest_run_dir",
    "load_snapshot",
    "marker_spans",
    "render_all",
    "sha256_file",
    "write_snapshot",
]
