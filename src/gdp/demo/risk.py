"""The attention heuristic: geometry only — no score, no level, no ranking, no number (H9).

Monocular BDD100K has no calibration and no depth ground truth, so nothing this module returns can
honestly be called a distance, a probability, or a graded assessment of danger. It returns a
*subset of the detector's own boxes*, unchanged — the only thing added is which ones to draw
differently, badged `HEURISTIC_BADGE` by the caller. The *measured* risk result for this project
lives in spec 08's DriveLM `planning`/`behavior` per-category rows (`specs/README.md`); this
function is a demo illustration that links to it, never a stand-in for it.
"""

from __future__ import annotations

from collections.abc import Sequence

from gdp.detect.predictions import Detection

# Centre-x in the middle third of the frame AND bottom edge below 55% of frame height — a crude
# "roughly ahead of the ego vehicle" proxy for a fixed front-facing camera. Both numbers are
# geometry-only constants, chosen by eye against the fixture scenes, not fit against any ground
# truth (there is none to fit against).
_MIDDLE_THIRD = (1.0 / 3.0, 2.0 / 3.0)
_BOTTOM_EDGE_FRACTION = 0.55


def heuristic_highlights(dets: Sequence[Detection], *, width: int, height: int) -> list[Detection]:
    """Boxes whose centre-x falls in the middle third of the frame and whose bottom edge sits
    below `_BOTTOM_EDGE_FRACTION` of the frame height — a crude "roughly ahead of the ego vehicle"
    proxy, and nothing more. Not a distance, not a probability, not a graded judgment of danger.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got ({width}, {height})")

    lo, hi = _MIDDLE_THIRD
    out = []
    for det in dets:
        x0, y0, x1, y1 = det.xyxy
        cx = (x0 + x1) / 2
        if lo * width <= cx <= hi * width and y1 >= _BOTTOM_EDGE_FRACTION * height:
            out.append(det)
    return out
