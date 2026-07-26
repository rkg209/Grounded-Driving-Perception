# Rendering the architecture diagram

`docs/architecture.mmd` is the **source of truth** for the architecture diagram, and it is what
`tests/test_report_architecture.py` audits (two disjoint model subgraphs, no edge between them, no
fusion/tracker/planner vocabulary). `README.md` embeds the same source verbatim in a ```` ```mermaid ````
block, which GitHub renders natively — so the README needs no image file to show the diagram.

A PNG is only needed for contexts that do not render mermaid (slides, a PDF CV, an email).

## The command

```bash
npx -y @mermaid-js/mermaid-cli -i docs/architecture.mmd -o docs/architecture.png -b white -w 1600
```

## Status: not rendered on this machine

`docs/architecture.png` **has not been generated**, and this repo does not claim it exists.
`mermaid-cli` pulls a headless Chromium on first use, which is a network download this project's
laptop workflow deliberately avoids (the same reason the demo video is a human step, see
`docs/demo-recording.md`).

If you render it, keep two things true, or the diagram stops matching what the tests guarantee:

1. Render **from `docs/architecture.mmd`** — never redraw it by hand in a diagramming tool. A
   hand-drawn box that adds an arrow from the detector to the VLM would imply one joint model
   feeding another (H6), and no test would catch it.
2. Re-render after any edit to `.mmd`, and update the README's embedded copy in the same change —
   `tests/test_report_architecture.py` asserts the two are byte-identical.
