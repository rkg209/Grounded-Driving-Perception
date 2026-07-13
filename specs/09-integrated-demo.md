# Spec 09 — Integrated demo

**Status:** draft · **Compute:** laptop · **Depends on:** 03 (or 05), 07

## Objective

One Gradio interface over the same driving scenes: **type a query → see boxes** (Stage 1), **ask a
question → see an answer** (Stage 2), and **generate a readable scene report** (presentation layer).
This is what gets recorded as the demo video and linked from the resume.

## Inputs / outputs

- **In:** the fine-tuned detector (spec 03, or the ONNX INT8 variant from 05) and the fine-tuned VLM
  (spec 07).
- **Out:** `gdp demo` launches a Gradio app; a recorded demo video for the write-up.

## Approach

1. Two panels, **side by side, visibly separate** — because they *are* two separate models (H6):
   - **left:** image + text query → boxes with confidences,
   - **right:** image + question → generated answer.
2. A visible caption in the UI itself: *"Two separate models. The detector does not feed the VLM."*
   The demo is where a joint-model impression would most easily form; kill it at the source.
3. **Scene report panel** (third panel) — composes the two models' outputs into a readable summary:

   ```
   Scene summary
     3 vehicles, 1 cyclist, 1 pedestrian detected.        [detector — Stage 1]
     A cyclist is approaching from the right.             [VLM — Stage 2]

   Attention
     Monitor the pedestrian near the crosswalk.           [VLM — Stage 2]

   ⚠ Qualitative demo. Composed from two separate models. Not a measured result.
   ```

   **Every line is attributed to the model that produced it.** This is what makes composition
   H6-safe: the report is a *presentation layer over two models*, not a third model, and the UI never
   lets you forget it. Attribution tags are non-negotiable — an unattributed composed report is
   exactly the joint-system illusion H6 exists to prevent.
4. **Risk overlay — clearly unmeasured.** A heuristic highlight (e.g. boxes near the ego lane) may be
   shown, badged **"heuristic illustration — no accuracy claim (H9)"**. The *measured* risk result
   lives in spec 08 (DriveLM planning/behaviour accuracy) and the demo links to it. Never display a
   risk score, a risk probability, or a "risk level: HIGH" without that badge — a confident-looking
   hazard label with no ground truth behind it is the single most misleading thing this UI could show.
3. Ship a set of preloaded BDD/nuScenes scenes so the demo works without upload, and allow upload.
4. Show the detector's **confidence scores and threshold slider** — an interviewer will move it, and
   seeing precision/recall trade off live is a better answer than any slide.
5. Runs on the M4 (INT8 ONNX detector; the 3B VLM in bf16 or 4-bit). If the VLM is too slow locally,
   host Stage 2 on HF Spaces and say so — **do not fake latency by pre-computing answers**.
6. Out-of-taxonomy queries are welcome in the demo but labelled **"demo, not a measured result"**
   in the UI (H7).

## Acceptance criteria

1. `gdp demo` launches; all three panels work end-to-end on a preloaded scene.
2. The "two separate models" caption is present in the UI.
3. **Every line of the scene report carries a model-attribution tag.** A test asserts no unattributed
   sentence can be emitted.
4. Any risk overlay carries the "heuristic — no accuracy claim" badge.
5. Out-of-taxonomy query results are visibly labelled as unmeasured demos.
6. Latency shown is real, measured at request time — never hardcoded.
7. A demo video is recorded showing all three panels and the threshold slider.

## Honesty contract

- **H6** — **the highest-risk spec for this rule.** A single "ask anything about this scene" box
  that silently routes to two models would imply a joint system. Keep the panels separate, attribute
  every report line, and say why.
- **H9** — the scene report and any risk overlay are **qualitative demos with no metric**. The moment
  a number appears next to them, they need ground truth they do not have.
- **H7** — a demo is not a result. Nothing in the UI may display a metric that was not produced by
  specs 02/03/04/05/08.
- **H3** — no language in the UI suggesting in-vehicle deployment.
- **H5** — the report says what is *present* and what to *watch*; it never issues a driving command
  ("brake now", "merge"). Description, not control.

## Out of scope

No real-time video stream (single-frame — **H5**), no tracking, no user accounts, no persistence.
