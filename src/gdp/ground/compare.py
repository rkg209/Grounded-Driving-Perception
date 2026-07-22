"""The zero-shot -> fine-tuned grounding-accuracy delta (design decision 9) — H8, mechanically
enforced, on the same pattern as `gdp.eval.compare.build_comparison`.

`build_grounding_comparison` refuses to emit unless both `metrics.json`s agree on
`phrases_sha256`, `num_phrases`, and `box_threshold`: a delta computed across two different phrase
sets, phrase counts, or thresholds isn't a measurement of fine-tuning, it's noise. Every emitted
comparison also carries `is_self_built_benchmark` and `caveat` (H7/H9), so the label travels with
the number into any table that quotes it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from gdp.ground.evaluate import CAVEAT

_SAME_SET_FIELDS = ("phrases_sha256", "num_phrases", "box_threshold")


def build_grounding_comparison(
    zeroshot: dict[str, Any], finetuned: dict[str, Any]
) -> dict[str, Any]:
    mismatched = [
        field for field in _SAME_SET_FIELDS if zeroshot.get(field) != finetuned.get(field)
    ]
    if mismatched:
        details = ", ".join(
            f"{field}: zeroshot={zeroshot.get(field)!r} vs finetuned={finetuned.get(field)!r}"
            for field in mismatched
        )
        raise ValueError(
            f"refusing to compare mismatched grounding runs ({details}) — a zero-shot -> "
            "fine-tuned grounding-accuracy delta is only honest on the identical frozen phrase "
            "set at the identical box_threshold (H8)"
        )

    types = sorted(set(zeroshot["per_type"]) | set(finetuned["per_type"]))
    per_type: dict[str, dict[str, float]] = {}
    regressions: list[str] = []
    for qualifier_type in types:
        before = zeroshot["per_type"].get(qualifier_type, {}).get("accuracy", float("nan"))
        after = finetuned["per_type"].get(qualifier_type, {}).get("accuracy", float("nan"))
        delta = after - before
        per_type[qualifier_type] = {"before": before, "after": after, "delta": delta}
        if not math.isnan(delta) and delta < 0:
            regressions.append(qualifier_type)

    return {
        "phrases_sha256": zeroshot["phrases_sha256"],
        "num_phrases": zeroshot["num_phrases"],
        "box_threshold": zeroshot["box_threshold"],
        "zeroshot_model_id": zeroshot.get("model_id"),
        "finetuned_model_id": finetuned.get("model_id"),
        "zeroshot_created": zeroshot.get("created"),
        "finetuned_created": finetuned.get("created"),
        "accuracy_zeroshot": zeroshot["overall"]["accuracy"],
        "accuracy_finetuned": finetuned["overall"]["accuracy"],
        "accuracy_delta": finetuned["overall"]["accuracy"] - zeroshot["overall"]["accuracy"],
        "positives_accuracy_zeroshot": zeroshot["positives"]["accuracy"],
        "positives_accuracy_finetuned": finetuned["positives"]["accuracy"],
        "negatives_accuracy_zeroshot": zeroshot["negatives"]["accuracy"],
        "negatives_accuracy_finetuned": finetuned["negatives"]["accuracy"],
        "per_type": per_type,
        "regressions": sorted(regressions),
        "is_self_built_benchmark": True,
        "caveat": CAVEAT,
    }


def write_grounding_comparison(comparison: dict[str, Any], *, out_dir: Path) -> Path:
    path = out_dir / "grounding_comparison.json"
    path.write_text(json.dumps(comparison, indent=2) + "\n")
    return path


def load_grounding_metrics(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
