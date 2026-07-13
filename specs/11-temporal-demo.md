# Spec 11 — Multi-frame temporal reasoning (OPTIONAL STRETCH, demo-only)

**Status:** draft (stretch) · **Compute:** laptop/cluster · **Depends on:** 07, 09
**Priority: LAST. This is the first thing cut if the schedule slips.** It sits *after* the spec-05
checkpoint and after the Stage-2 headline (08) for a reason: nothing here is claimable, and the
project is already complete and defensible without it.

## Objective

Show that the fine-tuned VLM can reason over **a short sequence of frames** rather than a single
image — "is the pedestrian approaching the road?", "is the vehicle ahead slowing?" — as a **labelled
qualitative demo with no metric**.

## Why this is demo-only, and why that is not a cop-out

Temporal reasoning is genuinely valuable, and it is also where this project could quietly turn back
into the trajectory-prediction project that was deliberately abandoned. Two hard facts:

1. **There is no official benchmark for it here.** DriveLM and nuScenes-QA are keyframe-based. Any
   temporal QA set would be **questions we wrote, scored against answers we wrote** — a self-invented
   benchmark, which is exactly what **H9** forbids. So it gets **no accuracy number**. Not a small
   one, not a caveated one. None.
2. **A metric would require a tracker**, and a tracker is a system module that **H5** excludes.

What we *can* honestly show: Qwen2.5-VL **natively accepts multiple images**, so N consecutive frames
can be fed straight to the model. **No tracker, no motion estimation, no new architecture** — the
temporal capability comes free with the model we already fine-tuned. That is a legitimate and
interesting demonstration, and it is the whole of this spec.

## Inputs / outputs

- **In:** N consecutive nuScenes frames (N ≈ 3–6, same camera, same scene) + a temporal question.
- **Out:** a natural-language answer, shown in the demo with a **"qualitative demo — unmeasured"**
  badge. Plus a short qualitative write-up: where multi-frame input visibly helps, and where it
  hallucinates motion that is not there (the interesting failure).

## Approach

1. Sample short frame sequences from nuScenes scenes already used in spec 06 (no new dataset).
2. Feed them as a multi-image message via Qwen2.5-VL's chat template. Watch the **image-token budget**
   — N frames multiplies it, and this is the practical limit on N. Record the limit you hit; it is a
   real engineering finding.
3. Add a "temporal" tab to the spec-09 demo.
4. Write ~10 qualitative examples, **including the failures**: single-frame models routinely confabulate
   motion, and a VLM given 5 frames may still answer from the last one alone. Probe that explicitly by
   feeding the *same frame* 5 times and seeing if it still claims motion. **That control experiment is
   the most interesting thing in this spec** — it tests whether the model is genuinely using time or
   just pattern-matching a plausible driving narrative.

## Acceptance criteria

1. Multi-frame input runs end-to-end through the fine-tuned VLM.
2. The demo tab exists and badges every output as unmeasured.
3. ≥10 qualitative examples documented, failures included.
4. The **repeated-identical-frame control** is run and its result reported honestly.
5. **`metrics.json` contains no accuracy number for this spec.** If one appears, this spec is wrong.

## Honesty contract

- **H9** — **no self-invented temporal benchmark, no temporal accuracy number.** Ever.
- **H5** — no tracking, no trajectory output, no planning. The model describes; it does not predict a
  path or command a manoeuvre.
- **H7** — this is a **capability demo, not a result**, and every appearance of it says so. In the
  README it lives under "Qualitative demos", never under "Results".

## Out of scope

Tracking, optical flow, motion models, video encoders, trajectory prediction, any temporal metric.

## Honest framing for the write-up / interview

> "Temporal reasoning is a demo, not a result: there is no official temporal-QA benchmark for these
> datasets, so I show the capability qualitatively — including a control showing how often the model
> claims motion when given identical frames — rather than inventing a benchmark I would also be
> grading. Proper temporal evaluation needs a labelled benchmark, and that is the next step."

That answer is stronger than a number nobody can check.
