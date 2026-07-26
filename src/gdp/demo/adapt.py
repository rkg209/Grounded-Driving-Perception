"""Two tiny adapter shims that let an in-memory (path-only) image reach both models.

Nothing else in the repo takes a bare image path all the way through to a detector/VLM call
without a dataset behind it; Gradio hands the demo a path anyway, so these shims build the minimal
`Sample`/`DriveLMRecord` each model's existing code already knows how to consume — no new
inference code (plan design decision 7).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from gdp.data.core import Sample
from gdp.data.drivelm import DriveLMRecord


def sample_from_image(path: str | Path, *, image_id: int = 0) -> Sample:
    """`load_probe_images`'s per-file body (`gdp.probe.openvocab`), extracted for one file.

    `boxes=()` by construction — never a ground-truth stand-in (mirrors the probe module's own
    rule: out-of-taxonomy/demo images have nothing to score against).
    """
    p = Path(path)
    with Image.open(p) as img:
        width, height = img.size
    return Sample(image_id=image_id, image_path=p, width=width, height=height, boxes=())


def record_from_image(path: str | Path, question: str, *, scenes_root: str | Path) -> DriveLMRecord:
    """A shim `DriveLMRecord` carrying exactly one CAM_FRONT view and the demo's question.

    Safe to feed into `generate_with_stats` because that function takes only
    `build_messages(record, num_views)[:1]` — the user turn — so `answer=""` is never read on the
    generate path (`gdp.vqa.generate:56`). `cfg.vlm.num_views` already defaults to 1 (CAM_FRONT),
    so a single uploaded image needs no multi-view handling.

    **This record is never written to a `.jsonl` and never scored** (H2) — it exists only to
    reuse `generate_with_stats`'s call signature for one live demo request.
    """
    p = Path(path).resolve()
    root = Path(scenes_root).resolve()
    try:
        image_ref = str(p.relative_to(root))
    except ValueError:
        # Gradio uploads land in a temp dir outside scenes_root — an absolute value still works:
        # `Path(nuscenes_root) / <absolute>` returns the absolute path unchanged (pathlib's join
        # semantics), so generate_with_stats resolves it correctly either way.
        image_ref = str(p)

    return DriveLMRecord(
        qa_id=f"demo-{p.stem}",
        scene_token="demo-scene",
        frame_token="demo-frame",
        category="demo",
        question=question,
        answer="",
        object_tags=[],
        tag=[],
        image_paths={"CAM_FRONT": image_ref},
        view_order=["CAM_FRONT"],
        official_split="demo",
        is_synthetic="fixtures" in p.parts,
    )
