"""Spec 10 task 8: the architecture diagram is an artifact under test, not a picture.

`claims-auditor.md`'s step 5 audits the diagram for implying one joint model (H6). A diagram is the
easiest place in a repo to overclaim silently — one arrow from the detector to the VLM and a reader
concludes Stage 1 feeds Stage 2 — so the mermaid *source* is asserted here, and the README embeds
that same source verbatim. No model is loaded, so this file is not `model_heavy`.
"""

from __future__ import annotations

import re

import pytest

from gdp.paths import repo_root

MMD = repo_root() / "docs" / "architecture.mmd"
README = repo_root() / "README.md"

STAGE1_NODES = ("A1", "A2", "A3")
STAGE2_NODES = ("B1", "B2", "B3")

# H4/H5/H6 vocabulary: none of this exists in the system, so none of it may appear in its diagram.
FORBIDDEN = ("fusion", "lidar", "radar", "tracker", "tracking", "planner", "joint", "end-to-end")


@pytest.fixture(scope="module")
def mmd() -> str:
    return MMD.read_text()


def _edges(text: str) -> list[tuple[str, str]]:
    """Every `X --> Y` edge, as bare node ids."""
    edges = []
    for line in text.splitlines():
        match = re.search(r"^\s*(\w+)(?:\[[^\]]*\])?\s*-->\s*(\w+)", line)
        if match:
            edges.append((match.group(1), match.group(2)))
    return edges


def test_diagram_declares_both_stages_as_separate_subgraphs(mmd):
    assert 'subgraph S1["Stage 1' in mmd
    assert 'subgraph S2["Stage 2' in mmd


def test_no_edge_crosses_between_the_two_model_subgraphs(mmd):
    """The H6 assertion. Stage 1's output is not Stage 2's input, in either direction."""
    for src, dst in _edges(mmd):
        crossing = (src in STAGE1_NODES and dst in STAGE2_NODES) or (
            src in STAGE2_NODES and dst in STAGE1_NODES
        )
        assert not crossing, f"edge {src} --> {dst} joins the two models into one pipeline (H6)"


def test_both_stages_feed_only_the_presentation_layer(mmd):
    edges = _edges(mmd)
    assert ("A3", "C1") in edges and ("B3", "C1") in edges


def test_presentation_layer_is_labelled_a_demo(mmd):
    assert "demo only, no accuracy number" in mmd


def test_each_stage_names_the_metric_it_is_measured_by(mmd):
    """A diagram that shows capability without naming its measurement is a marketing diagram."""
    assert "measured by mAP" in mmd
    assert "measured by DriveLM's official metric" in mmd


@pytest.mark.parametrize("word", FORBIDDEN)
def test_diagram_contains_no_forbidden_capability(mmd, word):
    assert word not in mmd.lower(), f"{word!r} appears in the architecture diagram"


def test_readme_embeds_the_mermaid_source_verbatim(mmd):
    """If these drift, the README shows a diagram no test audits."""
    match = re.search(r"```mermaid\n(.*?)```", README.read_text(), re.S)
    assert match is not None, "README has no mermaid architecture block"
    assert match.group(1) == mmd


def test_readme_states_the_two_models_are_separate():
    assert "**These are two separate models.**" in README.read_text()
