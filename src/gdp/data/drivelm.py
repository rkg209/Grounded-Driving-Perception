"""DriveLM's raw nested JSON -> one flat `DriveLMRecord` per QA pair, official split preserved.

DriveLM's top level keys a JSON object by 32-hex nuScenes scene **token**; each scene has
`key_frames` (keyed by frame token), each frame has `image_paths` (camera name -> relative image
path), `key_object_infos`, and `QA` (category -> list of `{"Q": ..., "A": ...}`). Answers embed
object-reference tags like `<c1,CAM_FRONT,1088.3,497.5>` **verbatim** — those tags are the official
ground-truth string DriveLM/spec 08 scores against, so `question`/`answer` are copied byte-for-byte;
nothing here rewrites or strips them (H2).

Every dropped QA pair is counted **by reason**, never silently swallowed (H7), mirroring
`bdd100k.py`'s `ConversionStats` pattern. Category keys are copied verbatim from DriveLM's own
JSON, including its "behavior" (US) spelling — see `gdp.config.DRIVELM_CATEGORIES`.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gdp.config import DRIVELM_CATEGORIES
from gdp.data.splits import OfficialSplits, resolve_split
from gdp.paths import git_sha

# The six nuScenes camera views, in a fixed order. A record must carry a path for every one of
# these (spec 06 design decision 5) — discarding the five non-front views would make QA pairs
# whose object tags reference them unanswerable *by construction*, a drop reason we would have
# invented ourselves.
VIEW_ORDER: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

DROP_REASONS = (
    "missing_qa_text",
    "unknown_category",
    "missing_image_path",
    "image_file_missing",
    "missing_tag",
)

_OBJECT_TAG_RE = re.compile(r"<([^,>]+),([^,>]+),(-?[\d.]+),(-?[\d.]+)>")


def parse_object_tags(text: str) -> list[dict[str, Any]]:
    """Parse `<c1,CAM_FRONT,1088.3,497.5>`-style tags out of an answer string.

    Returns a derived, analysis-only field (spec 08's failure analysis) — parsed once here rather
    than re-parsed with a regex in three places. The source `answer` string is never modified.
    """
    tags = []
    for ref, camera, x, y in _OBJECT_TAG_RE.findall(text):
        tags.append({"ref": ref, "camera": camera, "x": float(x), "y": float(y)})
    return tags


@dataclass(frozen=True)
class DriveLMRecord:
    """One QA pair, chat-formatting-ready. `question`/`answer` are byte-verbatim from DriveLM."""

    qa_id: str
    scene_token: str
    frame_token: str
    category: str
    question: str
    answer: str
    object_tags: list[dict[str, Any]]
    # DriveLM's own scorer-routing tag (e.g. [0], [2]) — NOT the same thing as `object_tags` above
    # (those are parsed <cX,CAM,x,y> references out of the answer text). This is what
    # third_party/drivelm/evaluation.py's `evaluation_suit.forward(tag, ...)` uses to route an item
    # to accuracy/chatgpt/language/match (spec 08 design decision 6) — it varies within a category,
    # so it cannot be inferred from `category` alone and must travel with the record verbatim (H2).
    tag: list[int]
    image_paths: dict[str, str]
    view_order: list[str]
    official_split: str
    is_synthetic: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConversionStats:
    """Drop accounting for one `convert_drivelm` call. Dataset statistics, not a metric (H9)."""

    scenes: int = 0
    frames: int = 0
    qa_total: int = 0
    qa_kept: int = 0
    dropped: dict[str, int] = field(default_factory=lambda: dict.fromkeys(DROP_REASONS, 0))

    def to_json(self) -> dict[str, Any]:
        return {
            "scenes": self.scenes,
            "frames": self.frames,
            "qa_total": self.qa_total,
            "qa_kept": self.qa_kept,
            "dropped": dict(self.dropped),
        }


def convert_drivelm(
    raw: dict[str, Any],
    *,
    scene_meta: dict[str, str],
    splits: OfficialSplits,
    images_root: str | Path,
    categories: tuple[str, ...] = DRIVELM_CATEGORIES,
    is_synthetic: bool,
) -> tuple[list[DriveLMRecord], list[DriveLMRecord], ConversionStats]:
    """Convert DriveLM's raw nested JSON into flat train/val record lists.

    `raw` is DriveLM's top-level dict: `{scene_token: {"key_frames": {frame_token: {...}}}}`.
    A scene token absent from `scene_meta`, or a scene name in neither official list, is a **hard
    error** propagated from `resolve_split` — never a silent drop (design decision 2): silently
    dropping scenes would shrink the split every later number is computed on.

    Deterministic: scenes/frames/categories/QA are all walked in sorted-key order, so re-running
    on the same input produces byte-identical output.
    """
    images_root = Path(images_root)
    stats = ConversionStats()
    train_records: list[DriveLMRecord] = []
    val_records: list[DriveLMRecord] = []

    for scene_token, scene in sorted(raw.items()):
        stats.scenes += 1
        split = resolve_split(scene_token, scene_meta=scene_meta, splits=splits)

        for frame_token, frame in sorted(scene.get("key_frames", {}).items()):
            stats.frames += 1
            image_paths: dict[str, str] = frame.get("image_paths", {})
            missing_views = set(VIEW_ORDER) - set(image_paths)
            file_missing = not missing_views and any(
                not (images_root / rel).is_file() for rel in image_paths.values()
            )

            qa_dict: dict[str, list[dict[str, str]]] = frame.get("QA", {})
            for category, qa_list in sorted(qa_dict.items()):
                for idx, qa in enumerate(qa_list):
                    stats.qa_total += 1
                    question = qa.get("Q", "")
                    answer = qa.get("A", "")
                    tag = qa.get("tag")

                    if category not in categories:
                        stats.dropped["unknown_category"] += 1
                        continue
                    if not question.strip() or not answer.strip():
                        stats.dropped["missing_qa_text"] += 1
                        continue
                    if not tag:
                        stats.dropped["missing_tag"] += 1
                        continue
                    if missing_views:
                        stats.dropped["missing_image_path"] += 1
                        continue
                    if file_missing:
                        stats.dropped["image_file_missing"] += 1
                        continue

                    record = DriveLMRecord(
                        qa_id=f"{scene_token}-{frame_token}-{category}-{idx:03d}",
                        scene_token=scene_token,
                        frame_token=frame_token,
                        category=category,
                        question=question,
                        answer=answer,
                        object_tags=parse_object_tags(answer),
                        tag=list(tag),
                        image_paths=dict(image_paths),
                        view_order=list(VIEW_ORDER),
                        official_split=split,
                        is_synthetic=is_synthetic,
                    )
                    stats.qa_kept += 1
                    (train_records if split == "train" else val_records).append(record)

    assert_no_scene_overlap(train_records, val_records)
    return train_records, val_records, stats


def assert_no_scene_overlap(
    train_records: list[DriveLMRecord], val_records: list[DriveLMRecord]
) -> None:
    """The spec's non-negotiable gate (H2): raise, listing the offending tokens, if any
    `scene_token` appears in both splits. Runs inside `convert_drivelm` itself, before a single
    line of JSONL is written — a corrupt split must never reach disk, not just never pass a test
    that could be skipped (design decision 3).
    """
    train_tokens = {r.scene_token for r in train_records}
    val_tokens = {r.scene_token for r in val_records}
    overlap = train_tokens & val_tokens
    if overlap:
        raise ValueError(
            f"scene-level leakage: {sorted(overlap)} appear in both train and val records"
        )


def write_jsonl(records: list[DriveLMRecord], path: str | Path) -> Path:
    """Write records sorted by `qa_id`, one JSON object per line. Deterministic: same records ->
    same bytes, always (the `train/logging_jsonl.py` writer, made deterministic)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in sorted(records, key=lambda r: r.qa_id):
            f.write(json.dumps(record.to_json()) + "\n")
    return path


def _category_counts(records: list[DriveLMRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.category] = counts.get(r.category, 0) + 1
    return counts


def _l1_drift(counts: dict[str, int], reference_props: dict[str, float]) -> float:
    total = sum(counts.values())
    if total == 0:
        return sum(reference_props.values())
    props = {c: counts.get(c, 0) / total for c in reference_props}
    return sum(abs(props[c] - reference_props[c]) for c in reference_props)


@dataclass(frozen=True)
class SubsampleResult:
    """`gdp data prepare-drivelm --train-fraction`'s output. **Training-only** (H7) — the caller
    must never apply this to val, which is why there is no `val_records` parameter anywhere in
    this module's subsampling path."""

    unit: str
    seed: int
    fraction: float
    scenes_before: int
    scenes_after: int
    qa_before: int
    qa_after: int
    category_distribution_before: dict[str, int]
    category_distribution_after: dict[str, int]
    l1_drift: float
    selected_scene_tokens: list[str]
    created: str
    git_sha: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def subsample_scenes(
    train_records: list[DriveLMRecord], *, fraction: float, seed: int, max_swap_passes: int = 50
) -> SubsampleResult:
    """Select a fraction of **train** scenes (never a fraction of a scene, never val — design
    decision 7), preserving the category distribution as closely as possible.

    Starts from a seeded random sample of whole scene tokens, then greedily swaps one selected
    scene for one unselected scene whenever the swap strictly reduces the category distribution's
    L1 drift against the full train set, until a full pass finds no improving swap or
    `max_swap_passes` is reached. Deterministic: same `(train_records, fraction, seed)` -> same
    `selected_scene_tokens`, always.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if not train_records:
        raise ValueError("subsample_scenes requires at least one train record")

    scene_tokens = sorted({r.scene_token for r in train_records})
    n_before = len(scene_tokens)
    k = max(1, round(fraction * n_before))
    k = min(k, n_before)

    full_counts = _category_counts(train_records)
    full_total = sum(full_counts.values())
    full_props = {c: full_counts.get(c, 0) / full_total for c in full_counts}

    if k >= n_before:
        selected = set(scene_tokens)
    else:
        records_by_scene: dict[str, list[DriveLMRecord]] = {t: [] for t in scene_tokens}
        for r in train_records:
            records_by_scene[r.scene_token].append(r)

        def drift_of(sel: set[str]) -> float:
            counts: dict[str, int] = {}
            for t in sel:
                for c, n in _category_counts(records_by_scene[t]).items():
                    counts[c] = counts.get(c, 0) + n
            return _l1_drift(counts, full_props)

        rng = random.Random(seed)
        selected = set(rng.sample(scene_tokens, k))
        current_drift = drift_of(selected)

        for _ in range(max_swap_passes):
            improved = False
            for out_tok in sorted(selected):
                for in_tok in sorted(set(scene_tokens) - selected):
                    candidate = (selected - {out_tok}) | {in_tok}
                    d = drift_of(candidate)
                    if d < current_drift - 1e-12:
                        selected, current_drift, improved = candidate, d, True
                        break
                if improved:
                    break
            if not improved:
                break

    selected_records = [r for r in train_records if r.scene_token in selected]
    after_counts = _category_counts(selected_records)

    return SubsampleResult(
        unit="scene",
        seed=seed,
        fraction=fraction,
        scenes_before=n_before,
        scenes_after=len(selected),
        qa_before=len(train_records),
        qa_after=len(selected_records),
        category_distribution_before=full_counts,
        category_distribution_after=after_counts,
        l1_drift=_l1_drift(after_counts, full_props),
        selected_scene_tokens=sorted(selected),
        created=datetime.now(UTC).isoformat(),
        git_sha=git_sha(),
    )


def write_subsample_json(result: SubsampleResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2) + "\n")
    return path
