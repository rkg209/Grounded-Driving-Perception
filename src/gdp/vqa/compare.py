"""The base -> fine-tuned VQA accuracy delta (spec 08 design decision 8) — H8, mechanically
enforced, mirroring `gdp.eval.compare.build_comparison`'s mismatch-raises pattern for Stage 1.

`build_vqa_comparison` refuses to emit unless both `metrics.json`s agree on the split, the item
count, the decoding settings, the scorer identity, and which sub-metrics were actually computed: a
delta computed across two different splits, decodings, or scorer versions isn't a measurement of
fine-tuning, it's noise dressed up as a result. `regressions` is a field in the returned dict, not
a paragraph someone has to remember to write (H7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gdp.config import DRIVELM_CATEGORIES

_SAME_RUN_FIELDS = ("val_jsonl_sha256", "num_items", "generation", "scorer_sha256")


def _submetric_names(node: dict[str, Any]) -> set[str]:
    return {s["name"] for s in node["submetrics"]["computed"]}


def _accuracy_score(node: dict[str, Any]) -> float | None:
    return next(s["score"] for s in node["submetrics"]["computed"] if s["name"] == "accuracy")


def _accuracy_delta(base_node: dict[str, Any], finetuned_node: dict[str, Any]) -> dict[str, Any]:
    before = _accuracy_score(base_node)
    after = _accuracy_score(finetuned_node)
    delta = (after - before) if (before is not None and after is not None) else None
    return {"before": before, "after": after, "delta": delta}


def build_vqa_comparison(base: dict[str, Any], finetuned: dict[str, Any]) -> dict[str, Any]:
    mismatched = [f for f in _SAME_RUN_FIELDS if base.get(f) != finetuned.get(f)]
    if mismatched:
        details = ", ".join(
            f"{f}: base={base.get(f)!r} vs finetuned={finetuned.get(f)!r}" for f in mismatched
        )
        raise ValueError(
            f"refusing to compare mismatched runs ({details}) — a base -> fine-tuned VQA delta "
            "is only honest across the identical split/item-count/generation-settings/scorer (H8)"
        )

    base_names = _submetric_names(base["overall"])
    finetuned_names = _submetric_names(finetuned["overall"])
    if base_names != finetuned_names:
        raise ValueError(
            "refusing to compare: submetrics.computed names differ "
            f"(base={sorted(base_names)}, finetuned={sorted(finetuned_names)})"
        )

    per_category: dict[str, Any] = {}
    regressions: list[str] = []
    for category in DRIVELM_CATEGORIES:
        entry = _accuracy_delta(base["per_category"][category], finetuned["per_category"][category])
        per_category[category] = entry
        if entry["delta"] is not None and entry["delta"] < 0:
            regressions.append(category)

    return {
        "val_jsonl_sha256": base["val_jsonl_sha256"],
        "num_items": base["num_items"],
        "generation": base["generation"],
        "scorer_sha256": base["scorer_sha256"],
        "final_score_composition": base["final_score_composition"],
        "final_score": None,
        "base_created": base.get("created"),
        "finetuned_created": finetuned.get("created"),
        "overall_accuracy": _accuracy_delta(base["overall"], finetuned["overall"]),
        "per_category_accuracy": per_category,
        "regressions": sorted(regressions),
    }


def write_vqa_comparison(comparison: dict[str, Any], *, out_dir: Path) -> Path:
    path = out_dir / "comparison.json"
    path.write_text(json.dumps(comparison, indent=2) + "\n")
    return path
