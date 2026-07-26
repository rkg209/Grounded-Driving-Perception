"""The Stage-2 sanity report: DriveLM dataset statistics, never a metric (H9).

Mirrors `gdp.data.stats.build_stats`'s shape (per-split counts, `source.sha256`, `git_sha`,
`created`) and adds the fields spec 06's status note requires on every emitted artifact:
`split_provenance` and `caveat` name the withheld-DriveLM-leaderboard-answers resolution
([SEQ-0056]) so nobody downstream mistakes this for the DriveLM challenge split.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gdp.data.drivelm import ConversionStats, DriveLMRecord
from gdp.data.stats import _sha256
from gdp.paths import git_sha

SPLIT_PROVENANCE = "nuscenes.utils.splits (vendored) — src/gdp/data/nuscenes_splits.json"
OFFICIAL_SPLIT = "nuscenes-v1.0-trainval"
CAVEAT = (
    "DriveLM-nuScenes' challenge val/test answers are withheld behind EvalAI. This split "
    "instead partitions DriveLM's answered train file by the official nuScenes v1.0-trainval "
    "scene lists (700 train / 150 val) — the canonical published nuScenes partition, applied "
    "scene-clean. It is NOT the DriveLM leaderboard split."
)


def build_drivelm_stats(
    records: list[DriveLMRecord],
    conv_stats: ConversionStats,
    *,
    split: str,
    source_file: Path,
    is_synthetic: bool,
) -> dict[str, Any]:
    """Assemble the sanity report for one split's surviving records.

    `conv_stats` covers the *whole* conversion (both splits combined) — included verbatim under
    `conversion_stats`, never re-attributed to just this split, so the drop counts stay honest
    about their real scope.
    """
    category_distribution: dict[str, int] = {}
    for r in records:
        category_distribution[r.category] = category_distribution.get(r.category, 0) + 1

    return {
        "split": split,
        "scenes": len({r.scene_token for r in records}),
        "qa_count": len(records),
        "category_distribution": category_distribution,
        "conversion_stats": conv_stats.to_json(),
        "source": {"file": source_file.name, "sha256": _sha256(source_file)},
        "git_sha": git_sha(),
        "created": datetime.now(UTC).isoformat(),
        "split_provenance": SPLIT_PROVENANCE,
        "official_split": OFFICIAL_SPLIT,
        "is_synthetic": is_synthetic,
        "caveat": CAVEAT,
    }
