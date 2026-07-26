"""Spec 08 — scoring `predictions.jsonl` against DriveLM's official split with the vendored scorer.

No metric arithmetic lives here beyond slicing predictions into subsets and reading the vendored
`evaluation_suit`'s own numbers back out (H2/H9, CLAUDE.md §2). `score_predictions` re-invokes the
scorer once per category plus once overall (design decision 6) — DriveLM's language sub-metrics
are corpus-level, so a per-category number can only come from re-scoring that category's subset,
never from averaging per-item scores.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gdp.config import DRIVELM_CATEGORIES, Config, VQAEvalConfig
from gdp.data.drivelm import DriveLMRecord
from gdp.paths import git_sha
from gdp.vqa.official import (
    OfficialScorerUnavailable,
    run_official_scorer,
    scorer_composition,
    scorer_sha256,
    to_official_format,
)
from gdp.vqa.predict import assert_complete

_ZERO_ITEMS_REASON = "0 items of this type in the split"


def _score_subset(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one subset (overall or one category) — `computed` = what we can honestly run
    (`accuracy`, `language`), `omitted` = what the vendored scorer fuses with a paid ChatGPT judge
    (`chatgpt`, `match` — see `gdp.vqa.official.run_official_scorer`'s docstring). `final_score` is
    `null` here unconditionally: both weighted components above are always omitted (design
    decision 7)."""
    items = to_official_format(predictions) if predictions else []
    if items:
        raw = run_official_scorer(items)
    else:
        raw = {
            "accuracy": None,
            "language": None,
            "n_accuracy_items": 0,
            "n_language_items": 0,
            "n_chatgpt_items": 0,
            "n_match_items": 0,
        }

    def _computed_entry(name: str, n_items: int, score: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {"name": name, "n_items": n_items, "score": score}
        if n_items == 0:
            entry["reason"] = _ZERO_ITEMS_REASON
        return entry

    computed = [
        _computed_entry("accuracy", raw["n_accuracy_items"], raw["accuracy"]),
        _computed_entry("language", raw["n_language_items"], raw["language"]),
    ]
    omitted = [
        {
            "name": "chatgpt",
            "n_items": raw["n_chatgpt_items"],
            "reason": "requires a paid, non-deterministic OpenAI API judge; not run (H2/H9)",
            "effect": "the official composite final_score cannot be computed",
        },
        {
            "name": "match",
            "n_items": raw["n_match_items"],
            "reason": (
                "eval_match is fused with the same paid ChatGPT judge inside the vendored "
                "scorer (eval_chatGPT called internally) and cannot be computed without it"
            ),
            "effect": "the official composite final_score cannot be computed",
        },
    ]
    return {
        "n_items": len(predictions),
        "submetrics": {"computed": computed, "omitted": omitted},
        "final_score": None,
    }


def score_predictions(
    predictions: list[dict[str, Any]], records: list[DriveLMRecord], *, cfg: VQAEvalConfig
) -> dict[str, Any]:
    """Acceptance criterion 3's gate, then overall + per-category scoring (design decision 6)."""
    assert_complete(predictions, records)
    overall = _score_subset(predictions)
    per_category = {
        category: _score_subset([p for p in predictions if p["category"] == category])
        for category in DRIVELM_CATEGORIES
    }
    return {"overall": overall, "per_category": per_category}


def annotate_per_item(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-item exact-match flag where it's honestly computable (tag 0, multiple-choice style);
    `None` everywhere else — DriveLM's language/match/chatgpt sub-metrics are corpus-level or
    judge-based and have no per-item score (design decision 6). `gdp.vqa.failures`'s fallback
    error rule reads `exact_match is False` when it's not `None`."""
    rows = []
    for row in predictions:
        exact_match = (row["prediction"] == row["gt_answer"]) if 0 in row.get("tag", []) else None
        rows.append({**row, "exact_match": exact_match})
    return rows


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_vqa_metrics(
    *,
    model_role: str,
    predictions: list[dict[str, Any]],
    records: list[DriveLMRecord],
    cfg: VQAEvalConfig,
    gdp_cfg: Config,
    val_jsonl_path: str | Path,
    caveat: str,
    split_provenance: str,
    is_synthetic: bool,
) -> dict[str, Any]:
    """Assemble one model's full `metrics.json`-shaped record — provenance first, per H2/H8/H9,
    mirroring `gdp.eval.metrics_io.build_metrics`'s shape for Stage 1."""
    scored = score_predictions(predictions, records, cfg=cfg)
    return {
        "model_role": model_role,
        "num_items": len(predictions),
        "category_distribution": dict(Counter(p["category"] for p in predictions)),
        "is_synthetic": is_synthetic,
        "val_jsonl_sha256": _sha256_file(val_jsonl_path),
        "generation": {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "num_beams": cfg.num_beams,
        },
        "scorer_sha256": scorer_sha256(),
        "final_score_composition": scorer_composition(),
        "split_provenance": split_provenance,
        "caveat": caveat,
        "comparable_to_leaderboard": False,
        "overall": scored["overall"],
        "per_category": scored["per_category"],
        "config": asdict(gdp_cfg),
        "git_sha": git_sha(),
        "created": datetime.now(UTC).isoformat(),
    }


def write_vqa_metrics(metrics: dict[str, Any], *, out_dir: Path) -> Path:
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return path


def write_per_item(
    rows: list[dict[str, Any]], *, out_dir: Path, name: str = "per_item.jsonl"
) -> Path:
    path = out_dir / name
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


__all__ = [
    "OfficialScorerUnavailable",
    "annotate_per_item",
    "build_vqa_metrics",
    "score_predictions",
    "write_per_item",
    "write_vqa_metrics",
]
