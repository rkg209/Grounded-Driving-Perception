# Recording the spec-09 demo video

This is the shot list for the video linked from the resume (spec 09 acceptance criterion 7). It is
a **human recording task**, not a code deliverable — `.claude/plans/09-integrated-demo.md`
explicitly lists "the recorded video file itself" under "Not in this plan". Nothing here is
narrated by `gdp`; it is a checklist for whoever presses record.

**Do not record the final video against zero-shot/base weights.** Today's `gdp demo` badges every
panel `zero-shot checkpoint — NOT the fine-tuned model (H8)` / `base VLM, no LoRA adapter loaded —
NOT the fine-tuned model (H8)`, honestly, because specs 03 and 07's real artifacts are still
cluster-blocked (see `specs/README.md`'s status notes). Once they land:

```
uv run gdp demo -c configs/default.yaml -c configs/demo.yaml \
  --checkpoint runs/03-finetune/<ts>/checkpoint-N \
  --adapter runs/07-finetune-vlm/<ts>/adapter
```

and confirm both badges flip from zero-shot/base to the real checkpoint ids *before* recording —
a demo video showing zero-shot weights labelled as the fine-tuned model would be the worst H8
violation in the repo. A dry run against today's zero-shot/base weights (`uv run gdp demo -c
configs/default.yaml -c configs/demo.yaml`) is fine for rehearsing the shot list itself.

## Shot list

1. **Open on the caption.** `SEPARATE_MODELS` ("Two separate models. The detector does not feed
   the VLM. Stage 1 = Grounding-DINO (boxes). Stage 2 = Qwen2.5-VL (text).") must be visible
   without scrolling, above both panels, before anything else happens.

2. **Panel 1 — detection, taxonomy mode.** Select `scene_000` (or a real BDD100K scene once
   available), leave the mode on `taxonomy`, click `Detect`. Show:
   - boxes with confidences drawn on the image,
   - the scene badge (`synthetic fixture — NOT real BDD100K` on a fixture scene),
   - the checkpoint badge naming the loaded weights,
   - the latency label reading `detector: N ms (measured, this request)`.

3. **Move the threshold slider** and let it settle. Show the box set changing **instantly** and
   the latency label switching to `re-filtered from the cached forward pass — no new inference` —
   this is the moment that proves no new inference ran.

4. **Panel 1 — free-text mode.** Switch to `free-text`, type a phrase outside the BDD100K
   taxonomy (e.g. `"traffic cone"`), click `Detect`. Show:
   - the mode note explaining the INT8-ONNX-is-prompt-frozen fork (`INT8 ONNX is exported with a
     frozen prompt ... free-text queries run the PyTorch detector`),
   - the `UNMEASURED_QUERY` tag (`demo, not a measured result — this phrase has no ground truth
     (H7)`).

5. **Panel 2 — VQA.** Pick a scene, ask a question from `configs/demo.yaml`'s `report_questions`
   (or a free one — Panel 2's box is free text, but it only ever talks to the VLM, never the
   detector). Show:
   - the answer text,
   - the backend badge (`Stage 2: local, bf16 on mps` or `Stage 2: remote (<host>) — hosted
     because the 3B VLM is slow on this laptop`),
   - the latency number, and ask a second question to show it's a **different** number each time
     (never a cached/precomputed answer).

6. **Panel 3 — scene report.** Click `Generate report`. Show, in the rendered text:
   - the `Scene summary` section ending every line in `[detector — Stage 1]` or `[VLM — Stage 2]`,
   - the `Attention` section ending its line in `[heuristic — no accuracy claim (H9)]`,
   - the heuristic overlay image, labelled with `HEURISTIC_BADGE`, showing **no** score, level, or
     percentage anywhere on it,
   - `REPORT_CAVEAT` (`⚠ Qualitative demo. Composed from two separate models. Not a measured
     result.`) at the bottom.

7. **Footer.** Show the hardware line and the artifact *paths* (`runs/05-deploy/<ts>/curve.png`,
   `runs/08-vqa/<ts>/metrics.json`) — no mAP, no accuracy figure rendered anywhere in the UI.

8. **Close on a full-window shot** with all three panels and the caption visible together, so a
   viewer who only watches the first and last five seconds still sees the H6 caption and the
   three-panel structure.

## What must never appear on camera

- A single input box that could be mistaken for routing to both models (there isn't one — this is
  a layout guarantee, not an editing instruction).
- `in-vehicle`, `production`, `real-time`, or `deployed in`, anywhere in the UI (asserted by
  `tests/test_demo_honesty.py`, but worth a visual sanity check before recording).
- A risk score, risk level, hazard probability, or a `%` next to the word "risk".
- Zero-shot/base weights presented as if they were the fine-tuned checkpoint/adapter.
