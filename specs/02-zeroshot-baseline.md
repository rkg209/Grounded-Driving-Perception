# Spec 02 — Zero-shot baseline (the bar)

**Status:** draft · **Compute:** cluster (full val); laptop for a small subset · **Depends on:** 01

## Objective

Run **pretrained, unmodified** Grounding-DINO on the BDD100K val split with class-name queries and
record its **zero-shot mAP**. This number is the bar every later Stage-1 claim is measured against
(H8). It is produced *before* any fine-tuning so it cannot be tuned toward.

## Inputs / outputs

- **In:** `IDEA-Research/grounding-dino-tiny` (no fine-tuning), BDD val (spec 01), the 10-class
  prompt.
- **Out:** `runs/02-zeroshot/<ts>/metrics.json` — mAP, mAP@50, per-class AP, **precision and recall
  at the operating threshold**, plus the config, checkpoint id, split name, image count, and git SHA.
  A metric without that provenance is a rumour (CLAUDE.md §7).

  *(Precision/recall are reported alongside mAP because they are what a non-detection specialist
  reads: mAP is threshold-free and abstract, while "at our operating point we catch X% of pedestrians
  and Y% of our pedestrian boxes are real" is the sentence an ADAS reviewer actually wants.)*

## Approach

1. `gdp detect` — batched inference: image + fixed class prompt → boxes, scores, per-box text span.
2. Map each predicted box back to a **class index via its text span**, using the inverse of spec
   01's mapping. (Grounding-DINO returns token spans, not class ids — this is where naive
   implementations silently mislabel everything.)
3. `gdp evaluate` — COCO-style mAP against BDD val GT. Use `pycocotools`-compatible evaluation so
   the number is comparable to published work rather than home-made.
4. Sweep `box_threshold` on a **held-out slice of train**, never on val — tuning a threshold on the
   eval split is leakage and would poison the very baseline we need to be trustworthy.
5. Record qualitative examples: what zero-shot misses (these become the "before" column of the
   Stage-1 story).

## Acceptance criteria

1. `metrics.json` contains mAP + per-class AP on the **full** official val split, with provenance.
2. Evaluation runs on the fixture (as a smoke path) and on real val (as the result path).
3. A test asserts the span→class mapping round-trips: a box whose span decodes to "traffic light"
   is assigned class index 8, not 0.
4. The threshold used for the headline number was chosen on train, and this is stated in the JSON.
5. mAP is plausible (a wildly-near-zero result means the prompt/span mapping is broken, not that the
   model is bad — investigate before reporting).

## Honesty contract

- **H1** — this is the pretrained model, untouched. Say so.
- **H8** — **this spec exists solely to create the baseline.** If it is skipped, spec 03's mAP is
  unclaimable. Do not "estimate" the zero-shot number from the paper: papers report COCO/LVIS, not
  BDD, and citing them as our baseline would be fabrication.
- **H7** — report per-class AP including the classes where zero-shot fails badly.

## Out of scope

No fine-tuning, no grounding phrases (spec 04), no deployment.
