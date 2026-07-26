"""Spec 10 tasks 5–6: marker handling, and the "generated, not typed" gate.

The load-bearing test in this file is `test_readme_generated_blocks_match_the_snapshot`: it is
acceptance criterion 1. Hand-editing a number inside a generated README block fails the default test
suite, in-process, in a clean clone — no subprocess, no network, no `runs/` directory needed.
"""

from __future__ import annotations

import pytest

from gdp.paths import repo_root
from gdp.report.inject import (
    BEGIN,
    END,
    MarkerError,
    check_blocks,
    extract_blocks,
    inject_blocks,
)
from gdp.report.snapshot import load_snapshot
from gdp.report.tables import BLOCK_NAMES, render_all

README = repo_root() / "README.md"


def _doc(*names: str) -> str:
    parts = ["# Title", "", "prose the renderer must never touch", ""]
    for name in names:
        parts += [BEGIN.format(name=name), "old body", END.format(name=name), ""]
    parts += ["trailing prose", ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------- markers


def test_extract_blocks_returns_each_body():
    assert extract_blocks(_doc("stage1-map", "grounding")) == {
        "stage1-map": ["old body"],
        "grounding": ["old body"],
    }


def test_inject_replaces_only_the_named_block():
    text = _doc("stage1-map", "grounding")
    out = inject_blocks(text, {"stage1-map": ["new", "body"]})
    assert extract_blocks(out) == {"stage1-map": ["new", "body"], "grounding": ["old body"]}
    assert "prose the renderer must never touch" in out
    assert "trailing prose" in out


def test_inject_is_idempotent():
    text = _doc("stage1-map")
    once = inject_blocks(text, {"stage1-map": ["a", "b"]})
    assert inject_blocks(once, {"stage1-map": ["a", "b"]}) == once


def test_inject_preserves_the_trailing_newline():
    text = _doc("stage1-map")
    assert inject_blocks(text, {"stage1-map": ["x"]}).endswith("\n")


def test_inject_raises_when_the_readme_has_no_such_marker_pair():
    with pytest.raises(MarkerError, match="grounding"):
        inject_blocks(_doc("stage1-map"), {"grounding": ["x"]})


def test_duplicate_begin_marker_raises():
    text = _doc("stage1-map") + _doc("stage1-map")
    with pytest.raises(MarkerError, match="duplicate"):
        extract_blocks(text)


def test_unterminated_block_raises():
    text = "\n".join(["# Title", BEGIN.format(name="grounding"), "body", ""])
    with pytest.raises(MarkerError, match="unterminated"):
        extract_blocks(text)


def test_end_without_begin_raises():
    text = "\n".join(["# Title", END.format(name="grounding"), ""])
    with pytest.raises(MarkerError, match="no BEGIN"):
        extract_blocks(text)


def test_check_blocks_is_empty_when_bodies_match():
    text = inject_blocks(_doc("stage1-map"), {"stage1-map": ["x"]})
    assert check_blocks(text, {"stage1-map": ["x"]}) == []


def test_check_blocks_names_the_drifted_block():
    text = inject_blocks(_doc("stage1-map"), {"stage1-map": ["x"]})
    assert check_blocks(text, {"stage1-map": ["y"]}) == ["stage1-map"]


# --------------------------------------------------------------------------- the gate


@pytest.fixture(scope="module")
def rendered():
    return render_all(load_snapshot())


def test_readme_has_every_generated_block(rendered):
    present = set(extract_blocks(README.read_text()))
    assert set(BLOCK_NAMES) <= present


def test_readme_generated_blocks_match_the_snapshot(rendered):
    """Acceptance criterion 1: no number in the README was typed by a human."""
    drifted = check_blocks(README.read_text(), rendered)
    assert drifted == [], (
        f"README blocks {drifted} differ from docs/metrics_snapshot.json — "
        "run `uv run gdp report render`; never edit inside a generated block"
    )


def test_hand_editing_a_generated_block_is_detected(rendered):
    """The negative half of the gate — without this, the test above could be passing vacuously."""
    tampered = README.read_text().replace("> **Not yet measured.**", "> **Measured:** 0.99 mAP.", 1)
    assert check_blocks(tampered, rendered) != []


def test_prose_outside_the_blocks_is_not_part_of_the_gate(rendered):
    """Editing hand-written prose must not fail the render check — only generated bodies are owned
    by the generator."""
    edited = README.read_text().replace(
        "## Failure analysis", "## Failure analysis (edited by a human)", 1
    )
    assert check_blocks(edited, rendered) == []
