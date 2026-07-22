"""The zero-shot -> fine-tuned mAP delta (spec 03 design decision 8) — H8, mechanically enforced.

`build_comparison` refuses to emit unless both `metrics.json`s agree on `dataset`, `split`,
`num_images`, and `box_threshold`: a delta computed across two different splits or thresholds isn't
a measurement of fine-tuning, it's noise dressed up as a result. `regressions` is a field in the
returned dict, not a paragraph someone has to remember to write (H7).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

_SAME_SPLIT_FIELDS = ("dataset", "split", "num_images", "box_threshold")


def build_comparison(zeroshot: dict[str, Any], finetuned: dict[str, Any]) -> dict[str, Any]:
    mismatched = [
        field for field in _SAME_SPLIT_FIELDS if zeroshot.get(field) != finetuned.get(field)
    ]
    if mismatched:
        details = ", ".join(
            f"{field}: zeroshot={zeroshot.get(field)!r} vs finetuned={finetuned.get(field)!r}"
            for field in mismatched
        )
        raise ValueError(
            f"refusing to compare mismatched runs ({details}) — a zero-shot -> fine-tuned mAP "
            "delta is only honest across the identical dataset/split/num_images/box_threshold (H8)"
        )

    classes = sorted(set(zeroshot["per_class_ap"]) | set(finetuned["per_class_ap"]))
    per_class: dict[str, dict[str, float]] = {}
    regressions: list[str] = []
    for cls in classes:
        before = zeroshot["per_class_ap"].get(cls, float("nan"))
        after = finetuned["per_class_ap"].get(cls, float("nan"))
        delta = after - before
        per_class[cls] = {"before": before, "after": after, "delta": delta}
        if not math.isnan(delta) and delta < 0:
            regressions.append(cls)

    return {
        "dataset": zeroshot["dataset"],
        "split": zeroshot["split"],
        "num_images": zeroshot["num_images"],
        "box_threshold": zeroshot["box_threshold"],
        "zeroshot_model_id": zeroshot.get("model_id"),
        "finetuned_model_id": finetuned.get("model_id"),
        "zeroshot_created": zeroshot.get("created"),
        "finetuned_created": finetuned.get("created"),
        "map_zeroshot": zeroshot["map"],
        "map_finetuned": finetuned["map"],
        "map_delta": finetuned["map"] - zeroshot["map"],
        "map50_zeroshot": zeroshot["map50"],
        "map50_finetuned": finetuned["map50"],
        "map50_delta": finetuned["map50"] - zeroshot["map50"],
        "per_class_ap": per_class,
        "regressions": sorted(regressions),
    }


def write_comparison(comparison: dict[str, Any], *, out_dir: Path) -> Path:
    path = out_dir / "comparison.json"
    path.write_text(json.dumps(comparison, indent=2) + "\n")
    return path


def load_metrics(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
