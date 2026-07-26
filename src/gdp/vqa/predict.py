"""Resumable prediction generation for spec 08 — GPU/cluster-expensive, CPU-cheap to resume.

`predict_split` is the only place spec 08 calls the model: it appends one JSON line per QA item to
`out_path`, skipping any `qa_id` already present so a SLURM wall-clock kill loses at most the
in-flight item, never the whole split (design decision 5). No score is computed here —
`gdp.vqa.score` (a separate command, design decision 2) does that, so this file can be re-run
against a partial or complete `predictions.jsonl` without ever touching the model again once
generation is done.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from gdp.config import VQAEvalConfig
from gdp.data.drivelm import DriveLMRecord
from gdp.vqa.generate import generate_with_stats


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Read a `predictions.jsonl` back into a list of row dicts. Missing file -> empty list, so a
    first `predict_split` call against a not-yet-created `out_path` doesn't need a special case."""
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def already_predicted(path: str | Path) -> set[str]:
    """The set of `qa_id`s already written to `path` — what `predict_split` resumes past."""
    return {row["qa_id"] for row in load_predictions(path)}


def assert_complete(predictions: list[dict[str, Any]], records: list[DriveLMRecord]) -> None:
    """Acceptance criterion 3: every val item, no more, no less. Raises naming both counts
    rather than silently scoring a partial run as if it were the split (H7)."""
    have = {row["qa_id"] for row in predictions}
    want = {r.qa_id for r in records}
    missing = want - have
    extra = have - want
    if missing or extra:
        raise ValueError(
            f"predictions incomplete: {len(missing)} missing qa_id(s) "
            f"(e.g. {sorted(missing)[:5]}), {len(extra)} extra qa_id(s) not in the split "
            f"(e.g. {sorted(extra)[:5]}) — want {len(want)} items, have {len(have)}"
        )


def predict_split(
    records: list[DriveLMRecord],
    processor: Any,
    model: Any,
    *,
    cfg: VQAEvalConfig,
    nuscenes_root: str | Path,
    device: torch.device,
    model_role: str,
    out_path: str | Path,
    limit: int | None = None,
) -> Path:
    """Generate an answer for every record not already in `out_path`, appending as it goes.

    `model_role` ("base" | "finetuned") is stamped onto every row so `gdp.vqa.score` can tell the
    two runs apart once their rows are merged or compared side by side. `limit` exists for smoke
    runs only — a `limit`-truncated file will fail `assert_complete`'s gate by design.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_predicted(out_path)

    todo = [r for r in records if r.qa_id not in done]
    if limit is not None:
        todo = todo[:limit]

    with out_path.open("a") as f:
        for record in todo:
            start = time.perf_counter()
            result = generate_with_stats(
                processor,
                model,
                record,
                nuscenes_root=nuscenes_root,
                device=device,
                num_views=1,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                num_beams=cfg.num_beams,
            )
            elapsed = time.perf_counter() - start
            row = {
                "qa_id": record.qa_id,
                "scene_token": record.scene_token,
                "frame_token": record.frame_token,
                "category": record.category,
                "question": record.question,
                "gt_answer": record.answer,
                "tag": record.tag,
                "prediction": result["text"],
                "model_role": model_role,
                "num_input_tokens": result["num_input_tokens"],
                "num_generated_tokens": result["num_generated_tokens"],
                "seconds": elapsed,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()

    return out_path
