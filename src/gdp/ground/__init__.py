"""Spec 04 — the self-built grounding evaluation set (CLAUDE.md §2, H9's sanctioned exception)."""

from __future__ import annotations

from gdp.ground.compare import (
    build_grounding_comparison,
    load_grounding_metrics,
    write_grounding_comparison,
)
from gdp.ground.evaluate import (
    PhraseResult,
    aggregate_results,
    ground_phrase,
    score_phrase,
    score_phrase_set,
)
from gdp.ground.phrases import (
    GtAnnotation,
    Phrase,
    PhraseSet,
    ValidationResult,
    assert_frozen,
    ensure_valid,
    freeze,
    load_gt_index,
    load_phrases,
    phrases_sha256,
    validate_phrases,
)
from gdp.ground.sample import FrameSample, render_overlay, sample_and_render, sample_frames

__all__ = [
    "FrameSample",
    "GtAnnotation",
    "Phrase",
    "PhraseResult",
    "PhraseSet",
    "ValidationResult",
    "aggregate_results",
    "assert_frozen",
    "build_grounding_comparison",
    "ensure_valid",
    "freeze",
    "ground_phrase",
    "load_grounding_metrics",
    "load_gt_index",
    "load_phrases",
    "phrases_sha256",
    "render_overlay",
    "sample_and_render",
    "sample_frames",
    "score_phrase",
    "score_phrase_set",
    "validate_phrases",
    "write_grounding_comparison",
]
