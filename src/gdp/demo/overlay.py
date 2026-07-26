"""In-memory box drawing over a `Detection` list — the demo's own overlay renderer.

`gdp.ground.sample.render_overlay` (`gdp.ground.sample:62`) is `GtAnnotation`-typed and writes to
disk; this module is `Detection`-typed (the detector's own predictions, not ground truth) and
returns a new in-memory `Image.Image` because Gradio wants a picture back, not a file path. The
duplication between the two modules is deliberate, not an oversight — they draw different types
for different callers, and forcing one shared function would either make `gdp.ground` accept
`Detection`s it has no business scoring, or make this module write files it doesn't need.
"""

from __future__ import annotations

from collections.abc import Container, Sequence

from PIL import Image, ImageDraw

from gdp.detect.predictions import Detection

BOX_COLOR = (0, 200, 0)
HIGHLIGHT_COLOR = (255, 140, 0)


def draw_boxes(
    image: Image.Image,
    dets: Sequence[Detection],
    classes: Sequence[str],
    *,
    highlight: Container[int] = (),
) -> Image.Image:
    """Draw every detection in `dets` on a copy of `image`.

    `highlight` is the set of *indices into `dets`* to draw in `HIGHLIGHT_COLOR` instead of
    `BOX_COLOR` — the heuristic-attention subset (`gdp.demo.risk.heuristic_highlights`) is exactly
    the caller's use case, but this function has no opinion on what `highlight` means.
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for i, det in enumerate(dets):
        color = HIGHLIGHT_COLOR if i in highlight else BOX_COLOR
        x0, y0, x1, y1 = det.xyxy
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        name = (
            classes[det.class_id] if 0 <= det.class_id < len(classes) else f"class {det.class_id}"
        )
        draw.text((x0, max(0, y0 - 12)), f"{name} {det.score:.2f}", fill=color)
    return out
