"""The sanity report: dataset statistics, never a metric (H9).

BDD is dominated by `car`; `train` is near-absent and may be zero in val. The all-10-non-zero
assertion is a *fixture* test; on real data a class below `min_boxes_for_eval` is flagged in
`rare_classes` — surfaced as "too rare to evaluate" rather than quietly dropped to flatter later
mAP (H7).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gdp.data.bdd100k import ConversionStats
from gdp.paths import git_sha, run_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_stats(
    coco: dict[str, Any],
    conv_stats: ConversionStats,
    *,
    split: str,
    source_file: Path,
    min_boxes_for_eval: int = 100,
) -> dict[str, Any]:
    """Assemble the sanity report from a converted COCO dict and its conversion stats."""
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    per_class = {name: 0 for name in id_to_name.values()}
    for ann in coco["annotations"]:
        per_class[id_to_name[ann["category_id"]]] += 1

    rare_classes = [name for name, count in per_class.items() if count < min_boxes_for_eval]

    return {
        "split": split,
        "images": conv_stats.images,
        "images_without_boxes": conv_stats.images_without_boxes,
        "boxes": conv_stats.boxes,
        "per_class": per_class,
        "dropped": dict(conv_stats.dropped),
        "clipped": conv_stats.clipped,
        "rare_classes": rare_classes,
        "min_boxes_for_eval": min_boxes_for_eval,
        "source": {"file": source_file.name, "sha256": _sha256(source_file)},
        "git_sha": git_sha(),
        "created": datetime.now(UTC).isoformat(),
    }


def write_stats(
    stats: dict[str, Any],
    spec: str = "01-data",
    *,
    out_dir: Path | None = None,
    filename: str = "stats.json",
) -> Path:
    """Write the report to `<out_dir or a fresh runs/<spec>/<timestamp>>/<filename>`.

    Pass `out_dir` to share one timestamped directory across multiple splits in a single
    `gdp data prepare` invocation; omit it to get a fresh `gdp.paths.run_dir(spec)`.
    """
    out_dir = out_dir or run_dir(spec)
    out_path = out_dir / filename
    out_path.write_text(json.dumps(stats, indent=2) + "\n")
    return out_path
