"""The attribution type and the one function allowed to turn model output into display text.

Design decision 1 (`.claude/plans/09-integrated-demo.md`): acceptance criterion 3 asks for a test
that "no unattributed sentence can be emitted". A test that greps rendered output is weaker than a
design where the illegal state doesn't type-check, so `Attribution` is a `StrEnum` and
`ReportLine.source` has no default — omitting it is a `TypeError`, not a silent blank tag.

`render_report` is the **only** function in `src/gdp/demo/` that turns model output into display
text; no other f-string anywhere in this package may build a report sentence (H6). Every line it
emits carries its source, always in the same `[source]` suffix, so a reader never has to guess
which model produced which sentence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from gdp.demo.honesty import REPORT_CAVEAT
from gdp.detect.predictions import Detection

# H5: the report describes what is present and what to watch — it never issues control. This is a
# first-token tripwire against the obvious failure, not a claim of natural-language understanding
# (plan design decision 9's "honest limit" note) — a test says so.
IMPERATIVE_COMMANDS = frozenset(
    {"brake", "merge", "stop", "accelerate", "steer", "turn", "yield", "overtake"}
)


class Attribution(StrEnum):
    """Which model produced a `ReportLine` — a type, not a string convention (H6)."""

    DETECTOR = "detector — Stage 1"
    VLM = "VLM — Stage 2"
    HEURISTIC = "heuristic — no accuracy claim (H9)"


def _first_token(text: str) -> str:
    words = text.strip().split()
    if not words:
        return ""
    return words[0].strip(".,!?:;\"'").lower()


@dataclass(frozen=True)
class ReportLine:
    """One sentence, permanently bound to the model that produced it.

    `source` has no default: `ReportLine("text")` is a `TypeError`, not a line missing its tag.
    """

    text: str
    source: Attribution

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("ReportLine.text must not be empty")
        if "[" in self.text:
            raise ValueError(
                f"ReportLine.text must not contain '[' (would let a caller smuggle a fake "
                f"attribution tag past render_report): {self.text!r}"
            )
        first = _first_token(self.text)
        if first in IMPERATIVE_COMMANDS:
            raise ValueError(
                f"ReportLine.text reads as a driving command ({first!r} is the first word) — "
                f"the demo describes the scene, it never issues control (H5): {self.text!r}"
            )


@dataclass(frozen=True)
class ReportSection:
    heading: str
    lines: tuple[ReportLine, ...]


def render_report(sections: Sequence[ReportSection]) -> str:
    """The only function that renders `ReportLine`s to text. Every line ends in `[source]`."""
    blocks: list[str] = []
    for section in sections:
        rendered_lines = [f"{line.text:<52} [{line.source.value}]" for line in section.lines]
        blocks.append("\n".join([f"## {section.heading}", *rendered_lines]))
    return "\n\n".join(blocks)


# Raw BDD100K class names → the plural word a Stage-1 count line would read most naturally with.
# Anything not listed falls back to "<name>s" — never invented driving-relevance groupings (that
# would be exactly the kind of scoring-our-own-rule H9 warns about), just plain English
# pluralization.
_IRREGULAR_PLURALS = {
    "bus": "buses",
}


def _plural(class_name: str, count: int) -> str:
    if count == 1:
        return class_name
    return _IRREGULAR_PLURALS.get(class_name, f"{class_name}s")


def summarize_detections(dets: Sequence[Detection], classes: Sequence[str]) -> ReportLine:
    """One Stage-1 line: per-class counts of everything above the current threshold.

    `dets` is expected pre-filtered by `filter_detections` (`gdp.demo.backends`) — this function
    only counts, it does not itself apply a threshold.
    """
    if not dets:
        return ReportLine("Nothing detected above the current threshold.", Attribution.DETECTOR)

    counts: dict[str, int] = {}
    for det in dets:
        name = (
            classes[det.class_id] if 0 <= det.class_id < len(classes) else f"class {det.class_id}"
        )
        counts[name] = counts.get(name, 0) + 1

    parts = [f"{count} {_plural(name, count)}" for name, count in sorted(counts.items())]
    return ReportLine(f"{', '.join(parts)} detected.", Attribution.DETECTOR)


def compose_report(
    det_lines: Sequence[ReportLine],
    vlm_lines: Sequence[ReportLine],
    heuristic_lines: Sequence[ReportLine],
) -> str:
    """The full three-part report (plan design decision 9): a `Scene summary` section (detector
    counts + VLM answers), an `Attention` section (the badged heuristic highlights), then
    `REPORT_CAVEAT` — never omitted, never conditional."""
    sections = [
        ReportSection("Scene summary", tuple(det_lines) + tuple(vlm_lines)),
        ReportSection("Attention", tuple(heuristic_lines)),
    ]
    return render_report(sections) + f"\n\n{REPORT_CAVEAT}"
