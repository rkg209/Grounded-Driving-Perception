"""README generated-block markers (spec 10 decision 4).

    <!-- BEGIN GENERATED: stage1-map -->
    … rendered table …
    <!-- END GENERATED: stage1-map -->

Everything outside a marker pair is hand-written prose and is never touched. Everything inside is
owned by `gdp report render` and is overwritten wholesale.

`extract_blocks` **raises** on a missing, duplicated, unbalanced, or out-of-order pair rather than
skipping it. A silently-skipped block is exactly how a stale table survives a re-render and ends up
in front of a reviewer.
"""

from __future__ import annotations

import re

BEGIN = "<!-- BEGIN GENERATED: {name} -->"
END = "<!-- END GENERATED: {name} -->"

_MARKER_RE = re.compile(r"^<!-- (BEGIN|END) GENERATED: ([a-z0-9-]+) -->$", re.MULTILINE)


class MarkerError(ValueError):
    """A README whose generated-block markers cannot be trusted."""


def marker_spans(text: str) -> dict[str, tuple[int, int]]:
    """name -> (line index of BEGIN, line index of END), validated. Public because the README
    honesty tests need to know which lines the generator owns and which are hand-written."""
    lines = text.splitlines()
    opened: dict[str, int] = {}
    spans: dict[str, tuple[int, int]] = {}
    stack: list[str] = []

    for i, line in enumerate(lines):
        match = _MARKER_RE.match(line.strip())
        if match is None:
            continue
        kind, name = match.group(1), match.group(2)
        if kind == "BEGIN":
            if name in opened or name in spans:
                raise MarkerError(f"duplicate BEGIN marker for block {name!r} (line {i + 1})")
            opened[name] = i
            stack.append(name)
        else:
            if not stack or stack[-1] != name:
                raise MarkerError(
                    f"unbalanced END marker for block {name!r} (line {i + 1}) — "
                    f"expected END for {stack[-1]!r}"
                    if stack
                    else f"END marker for block {name!r} (line {i + 1}) with no BEGIN"
                )
            stack.pop()
            spans[name] = (opened.pop(name), i)

    if stack:
        raise MarkerError(f"unterminated generated block(s): {sorted(stack)}")
    return spans


def extract_blocks(text: str) -> dict[str, list[str]]:
    """The current body of each generated block, exactly as it sits in the file."""
    lines = text.splitlines()
    return {name: lines[start + 1 : end] for name, (start, end) in marker_spans(text).items()}


def inject_blocks(text: str, blocks: dict[str, list[str]]) -> str:
    """Replace each named block's body. Every block in `blocks` must already have its marker pair
    in `text` — `render` does not invent sections, it fills the ones the author placed."""
    spans = marker_spans(text)
    missing = sorted(set(blocks) - set(spans))
    if missing:
        raise MarkerError(
            f"README has no marker pair for generated block(s) {missing} — add "
            + ", ".join(f"`{BEGIN.format(name=m)}` / `{END.format(name=m)}`" for m in missing)
        )

    lines = text.splitlines()
    starts = {start: name for name, (start, _) in spans.items() if name in blocks}
    out: list[str] = []
    i = 0
    while i < len(lines):
        out.append(lines[i])
        if i in starts:
            name = starts[i]
            out.extend(blocks[name])
            i = spans[name][1]  # jump the old body; the END marker line is appended next
            out.append(lines[i])
        i += 1

    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + trailing_newline


def check_blocks(text: str, blocks: dict[str, list[str]]) -> list[str]:
    """Names of the blocks whose committed body differs from what the snapshot renders — i.e. the
    blocks somebody hand-edited, or that a newer snapshot has moved past. Empty list = clean."""
    current = extract_blocks(text)
    drifted = [name for name in sorted(blocks) if current.get(name) != blocks[name]]
    drifted += [name for name in sorted(blocks) if name not in current]
    return sorted(set(drifted))
