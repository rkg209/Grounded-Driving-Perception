"""Spec 09 — the integrated Gradio demo. Re-exports the honesty-load-bearing names so callers
never need to remember which submodule a constant or the attribution type lives in."""

from __future__ import annotations

from gdp.demo.app import build_app
from gdp.demo.honesty import (
    BASE_VLM_WEIGHTS,
    HEURISTIC_BADGE,
    REPORT_CAVEAT,
    SEPARATE_MODELS,
    SYNTHETIC_SCENE,
    UNMEASURED_QUERY,
    ZEROSHOT_WEIGHTS,
)
from gdp.demo.report import Attribution, ReportLine, ReportSection, render_report

__all__ = [
    "BASE_VLM_WEIGHTS",
    "HEURISTIC_BADGE",
    "REPORT_CAVEAT",
    "SEPARATE_MODELS",
    "SYNTHETIC_SCENE",
    "UNMEASURED_QUERY",
    "ZEROSHOT_WEIGHTS",
    "Attribution",
    "ReportLine",
    "ReportSection",
    "build_app",
    "render_report",
]
