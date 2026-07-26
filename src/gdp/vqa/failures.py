"""Spec 08 — the failure-analysis deliverable: a deterministic sample, machine-scaffolded, with
the human commentary fields left empty for a person to fill in (H7 — a real failure analysis, not
an LLM grading its own homework; design decision 11 explicitly rejects auto-generating the
commentary).

`sample_failures` needs each `per_item` row to already carry both models' predictions and an
`image_path` (the DriveLM record's `CAM_FRONT` path, resolved by the caller against its own
`DriveLMRecord`s — this module stays decoupled from that lookup so it's testable on plain dicts).
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

from gdp.vqa.hallucination import ungrounded_tags


def _is_error(row: dict[str, Any], error_on: str | None) -> bool:
    """True if this item counts as a failure to sample from. `error_on` names a per-item boolean
    field on the row (e.g. `"finetuned_exact_match"`) when per-item scores exist; falls back to a
    plain prediction-vs-GT string comparison otherwise — DriveLM's language/match/chatgpt
    sub-metrics are corpus-level and have no per-item score to fall back on (design decision 11)."""
    if error_on and row.get(error_on) is not None:
        return row[error_on] is False
    return row["finetuned_prediction"].strip() != row["gt_answer"].strip()


def sample_failures(
    per_item: list[dict[str, Any]],
    *,
    seed: int,
    n: int,
    categories: tuple[str, ...],
    error_on: str | None = None,
) -> list[dict[str, Any]]:
    """Deterministic (same `seed` -> same sample), stratified across `categories` so a category
    with fewer errors than its quota doesn't crowd out planning/behaviour — each category gets an
    even share of `n` first, and any shortfall is topped up from the remaining error pool."""
    rng = random.Random(seed)
    errors = [row for row in per_item if _is_error(row, error_on)]

    per_category_quota = max(1, n // len(categories))
    sampled: list[dict[str, Any]] = []
    for category in categories:
        cat_errors = sorted(
            (row for row in errors if row["category"] == category), key=lambda r: r["qa_id"]
        )
        rng.shuffle(cat_errors)
        sampled.extend(cat_errors[:per_category_quota])

    sampled_ids = {row["qa_id"] for row in sampled}
    remaining = sorted(
        (row for row in errors if row["qa_id"] not in sampled_ids), key=lambda r: r["qa_id"]
    )
    rng.shuffle(remaining)
    while len(sampled) < n and remaining:
        sampled.append(remaining.pop())

    return sampled[:n]


_HEADER = """# Failure analysis

**Generated fields** (machine-scaffolded by `gdp vqa failures`): image, question, GT answer, base
prediction, fine-tuned prediction, automatic ungrounded-tag flags.
**Hand-written fields** (left empty for a human): `Category:`, `Comment:`. An LLM-written
commentary on an LLM's own failures would be exactly the unfalsifiable artifact H7 asks this repo
not to produce (spec 08 design decision 11) — these are filled in by a person, or not at all.

**Error rule used:** {error_rule}
"""


def write_failures_md(
    samples: list[dict[str, Any]],
    *,
    out_dir: Path,
    images_root: str | Path,
    error_rule: str = "prediction != GT (no per-item score available for this sub-metric)",
) -> Path:
    """Copy each sample's `CAM_FRONT` image into `out_dir/failures/` and emit `failures.md`."""
    images_root = Path(images_root)
    images_dir = out_dir / "failures"
    images_dir.mkdir(parents=True, exist_ok=True)

    lines = [_HEADER.format(error_rule=error_rule)]
    for i, row in enumerate(samples, start=1):
        src_image = images_root / row["image_path"]
        dst_name = f"{i:03d}_{Path(row['image_path']).name}"
        dst_image = images_dir / dst_name
        if src_image.is_file():
            shutil.copy2(src_image, dst_image)

        base_ungrounded = ungrounded_tags(row["base_prediction"], row["gt_answer"])
        finetuned_ungrounded = ungrounded_tags(row["finetuned_prediction"], row["gt_answer"])

        lines.append(f"## {i}. `{row['qa_id']}` ({row['category']})\n")
        lines.append(f"![]({Path('failures') / dst_name})\n")
        lines.append(f"**Question:** {row['question']}\n")
        lines.append(f"**GT answer:** {row['gt_answer']}\n")
        lines.append(f"**Base prediction:** {row['base_prediction']!r}\n")
        lines.append(f"**Fine-tuned prediction:** {row['finetuned_prediction']!r}\n")
        lines.append(
            f"**Ungrounded tags (automatic):** base={len(base_ungrounded)}, "
            f"finetuned={len(finetuned_ungrounded)}\n"
        )
        lines.append("**Category:** \n")
        lines.append("**Comment:** \n")

    path = out_dir / "failures.md"
    path.write_text("\n".join(lines) + "\n")
    return path
