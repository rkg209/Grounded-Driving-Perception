"""Every UI-facing honesty string, as module-level constants — no logic.

`claims-auditor` greps *files*, not a running app (spec 09 design decision 2). The repo already
has this idiom — `CAVEAT` at `gdp.ground.evaluate:190`, `gdp.vqa.hallucination:16`,
`gdp.data.drivelm_stats:21` — and spec 09, the single highest-H6-risk spec in the repo, follows it
for every sentence the demo puts on screen. Nothing in `src/gdp/demo/` may build one of these
strings inline; it imports the constant from here, so a single `grep -rn` over this one file is
the whole audit surface for what the UI claims.
"""

from __future__ import annotations

SEPARATE_MODELS = (
    "Two separate models. The detector does not feed the VLM. "
    "Stage 1 = Grounding-DINO (boxes). Stage 2 = Qwen2.5-VL (text)."
)
REPORT_CAVEAT = "⚠ Qualitative demo. Composed from two separate models. Not a measured result."
HEURISTIC_BADGE = "heuristic illustration — no accuracy claim (H9)"
UNMEASURED_QUERY = "demo, not a measured result — this phrase has no ground truth (H7)"
SYNTHETIC_SCENE = "synthetic fixture — NOT real BDD100K"
ZEROSHOT_WEIGHTS = "zero-shot checkpoint — NOT the fine-tuned model (H8)"
BASE_VLM_WEIGHTS = "base VLM, no LoRA adapter loaded — NOT the fine-tuned model (H8)"
