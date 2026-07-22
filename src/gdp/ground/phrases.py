"""The grounding phrase schema, its loader, and the validator that is this spec's real risk.

A phrase is a referring expression over exactly one ground-truth box (or, for negatives, a claim
that a class is absent from a frame). `target_ann_id` — not a pasted `target_box` — is the source of
truth (design decision 3): every positive phrase names the **COCO annotation id** of its referent,
and "no orphans" becomes a lookup against the real annotations file rather than a float comparison
that can drift out of sync. `validate_phrases` runs every mechanical check this spec's honesty
depends on: orphan targets, negatives that are secretly true, missing qualifier-type coverage, and
the token-span round-trip (design decision 2) that keeps `evaluate.py`'s single-phrase grounding
from silently mis-scoring a phrase whose tokenization doesn't round-trip cleanly.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

from gdp.data.prompt import PromptSpans, get_tokenizer
from gdp.eval.operating_point import box_iou
from gdp.paths import git_sha, resolve

QUALIFIER_TYPES = ("spatial", "attribute", "relational", "negative")


@dataclass(frozen=True)
class Phrase:
    """One grounding phrase. `target_box` is *derived* from the annotations file at validation
    time (design decision 3) and stored purely for readability — it is never the source of truth."""

    image_id: int
    phrase: str
    qualifier_type: str
    author: str
    date: str
    target_ann_id: int | None = None
    target_box: tuple[float, float, float, float] | None = None
    absent_class: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "phrase": self.phrase,
            "qualifier_type": self.qualifier_type,
            "author": self.author,
            "date": self.date,
            "target_ann_id": self.target_ann_id,
            "target_box": list(self.target_box) if self.target_box is not None else None,
            "absent_class": self.absent_class,
        }


@dataclass(frozen=True)
class PhraseSet:
    phrases: tuple[Phrase, ...]
    is_synthetic: bool

    def __len__(self) -> int:
        return len(self.phrases)

    def per_type_counts(self) -> dict[str, int]:
        counts = Counter(p.qualifier_type for p in self.phrases)
        return {t: counts.get(t, 0) for t in QUALIFIER_TYPES}


@dataclass(frozen=True)
class GtAnnotation:
    ann_id: int
    image_id: int
    class_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]


def load_gt_index(
    annotations_path: str | Path,
) -> tuple[dict[int, GtAnnotation], dict[int, list[int]], dict[str, int]]:
    """COCO-style annotations JSON -> `(anns_by_id, ann_ids_by_image, class_id_by_name)`.

    Kept separate from `gdp.data.core.load_dataset`: that loader discards annotation ids, which
    this spec needs as the source of truth for `target_ann_id` (design decision 3).
    """
    path = resolve(annotations_path)
    raw = json.loads(path.read_text())
    cat_name_by_id = {c["id"]: c["name"] for c in raw["categories"]}
    class_id_by_name = {name: cid for cid, name in cat_name_by_id.items()}

    anns_by_id: dict[int, GtAnnotation] = {}
    ann_ids_by_image: dict[int, list[int]] = {}
    for a in raw["annotations"]:
        x, y, w, h = a["bbox"]
        ann = GtAnnotation(
            ann_id=a["id"],
            image_id=a["image_id"],
            class_id=a["category_id"],
            class_name=cat_name_by_id[a["category_id"]],
            xyxy=(x, y, x + w, y + h),
        )
        anns_by_id[ann.ann_id] = ann
        ann_ids_by_image.setdefault(ann.image_id, []).append(ann.ann_id)

    return anns_by_id, ann_ids_by_image, class_id_by_name


def load_phrases(path: str | Path) -> PhraseSet:
    """Load `phrases.json`. Phrase text is normalized to lowercase here (design decision 2) —
    the round-trip check and eval-time prompting both assume this normalization already happened."""
    raw = json.loads(resolve(path).read_text())
    phrases = tuple(
        Phrase(
            image_id=p["image_id"],
            phrase=p["phrase"].strip().lower(),
            qualifier_type=p["qualifier_type"],
            author=p["author"],
            date=p["date"],
            target_ann_id=p.get("target_ann_id"),
            target_box=tuple(p["target_box"]) if p.get("target_box") is not None else None,
            absent_class=p.get("absent_class"),
        )
        for p in raw["phrases"]
    )
    return PhraseSet(phrases=phrases, is_synthetic=bool(raw.get("is_synthetic", False)))


def phrases_sha256(path: str | Path) -> str:
    """Hash of the phrases file's raw bytes — any single-character post-freeze edit changes this
    (design decision 6)."""
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    type_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_phrases(
    phrase_set: PhraseSet,
    annotations_path: str | Path,
    *,
    min_phrases: int,
    qualifier_types: tuple[str, ...] = QUALIFIER_TYPES,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> ValidationResult:
    """Every mechanical guard this spec's honesty depends on. Returns errors (which must block a
    freeze) and ambiguity warnings (flagged, per design decision 4, not auto-fixed)."""
    anns_by_id, ann_ids_by_image, class_id_by_name = load_gt_index(annotations_path)
    tokenizer = tokenizer or get_tokenizer()

    result = ValidationResult(type_counts=phrase_set.per_type_counts())

    unknown_types = {p.qualifier_type for p in phrase_set.phrases} - set(QUALIFIER_TYPES)
    if unknown_types:
        result.errors.append(f"unknown qualifier_type(s): {sorted(unknown_types)}")

    missing_types = set(qualifier_types) - {p.qualifier_type for p in phrase_set.phrases}
    if missing_types:
        result.errors.append(f"missing qualifier_type coverage: {sorted(missing_types)}")

    if not phrase_set.is_synthetic and len(phrase_set) < min_phrases:
        result.errors.append(
            f"only {len(phrase_set)} phrases, need >= {min_phrases} (grounding.min_phrases)"
        )

    for i, p in enumerate(phrase_set.phrases):
        where = f"phrase[{i}] image_id={p.image_id} {p.phrase!r}"

        if p.qualifier_type == "negative":
            if p.target_ann_id is not None:
                result.errors.append(f"{where}: negative phrase must not carry a target_ann_id")
            if not p.absent_class:
                result.errors.append(f"{where}: negative phrase missing absent_class")
            elif p.absent_class in class_id_by_name:
                absent_id = class_id_by_name[p.absent_class]
                present = any(
                    anns_by_id[aid].class_id == absent_id
                    for aid in ann_ids_by_image.get(p.image_id, [])
                )
                if present:
                    result.errors.append(
                        f"{where}: absent_class {p.absent_class!r} IS present in image "
                        f"{p.image_id} — not a valid negative"
                    )
            else:
                result.errors.append(
                    f"{where}: absent_class {p.absent_class!r} is not a known class"
                )
        else:
            if p.target_ann_id is None:
                result.errors.append(f"{where}: positive phrase missing target_ann_id")
            elif p.target_ann_id not in anns_by_id:
                result.errors.append(f"{where}: target_ann_id {p.target_ann_id} does not exist")
            else:
                target = anns_by_id[p.target_ann_id]
                if target.image_id != p.image_id:
                    result.errors.append(
                        f"{where}: target_ann_id {p.target_ann_id} belongs to image "
                        f"{target.image_id}, not {p.image_id}"
                    )
                else:
                    for aid in ann_ids_by_image.get(p.image_id, []):
                        if aid == p.target_ann_id:
                            continue
                        other = anns_by_id[aid]
                        if (
                            other.class_id == target.class_id
                            and box_iou(other.xyxy, target.xyxy) >= 0.5
                        ):
                            result.ambiguous.append(
                                f"{where}: target_ann_id {p.target_ann_id} has a near-duplicate "
                                f"same-class GT box (ann_id {aid}, IoU>=0.5) in the same frame"
                            )

        try:
            spans = PromptSpans.from_classes((p.phrase,), tokenizer=tokenizer)
            decoded = spans.decode_span(0, tokenizer=tokenizer)
        except ValueError as exc:
            result.errors.append(f"{where}: token-span construction failed: {exc}")
            continue
        if decoded != p.phrase:
            result.errors.append(
                f"{where}: round-trip failed — decoded {decoded!r}, expected {p.phrase!r}"
            )

    return result


def ensure_valid(result: ValidationResult) -> None:
    if not result.ok:
        raise ValueError(
            "phrase validation failed:\n" + "\n".join(f"  - {e}" for e in result.errors)
        )


def freeze(phrases_path: str | Path, *, lock_path: str | Path, author: str) -> dict[str, Any]:
    """Write `phrases.lock.json`. `gdp ground evaluate` refuses to run unless the live file still
    hashes to this value (design decision 6) — held-out discipline becomes checkable, not
    asserted."""
    phrase_set = load_phrases(phrases_path)
    lock = {
        "sha256": phrases_sha256(phrases_path),
        "num_phrases": len(phrase_set),
        "per_type_counts": phrase_set.per_type_counts(),
        "frozen_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "author": author,
    }
    resolve(lock_path).write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def assert_frozen(phrases_path: str | Path, lock_path: str | Path) -> dict[str, Any]:
    """Raise unless `phrases_path` hashes to the value recorded in `lock_path` — the mechanical
    half of "no phrase was edited after seeing a model's prediction on it" (acceptance 5)."""
    lock_file = resolve(lock_path)
    if not lock_file.is_file():
        raise RuntimeError(f"no lock file at {lock_file} — run `gdp ground freeze` first")
    lock = json.loads(lock_file.read_text())
    live_hash = phrases_sha256(phrases_path)
    if live_hash != lock["sha256"]:
        raise RuntimeError(
            f"{resolve(phrases_path)} has changed since it was frozen "
            f"(live sha256={live_hash}, locked sha256={lock['sha256']}) — refusing to run. "
            "Editing a phrase after seeing a prediction breaks held-out discipline (H9's condition "
            "for this benchmark's exception); re-run `gdp ground freeze` only if the edit predates "
            "any evaluation run."
        )
    return lock
