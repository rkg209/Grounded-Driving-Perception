# Spec 02 — Zero-shot baseline (the bar)

**Status:** in-progress (all 7 laptop tasks done; the real H8 baseline awaits the cluster run —
`sbatch scripts/slurm/zeroshot_eval.slurm`, see acceptance 1) · **Compute:** cluster (full val);
laptop for a small subset · **Depends on:** 01

## Objective

Run **pretrained, unmodified** Grounding-DINO on the BDD100K val split with class-name queries and
record its **zero-shot mAP**. This number is the bar every later Stage-1 claim is measured against
(H8). It is produced *before* any fine-tuning so it cannot be tuned toward.

## Inputs / outputs

- **In:** `IDEA-Research/grounding-dino-tiny` (no fine-tuning), BDD val (spec 01), the 10-class
  prompt.
- **Out:** `runs/02-zeroshot/<ts>/metrics.json` — mAP, mAP@50, per-class AP, **precision and recall
  at the operating threshold**, plus the config, checkpoint id, split name, image count, and git SHA.
  A metric without that provenance is a rumour (CLAUDE.md §8).

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

## Tasks

- [x] **1. Dependency + detector skeleton.** Add `pycocotools` to `pyproject.toml`. Create
      `src/gdp/detect/__init__.py`, `src/gdp/detect/detector.py` with model/processor load
      (`gdp.seed.select_device`, cuda→mps→cpu), config wiring (`disable_custom_kernels`), and
      `src/gdp/detect/predictions.py` (`Detection` dataclass). No inference logic yet.
      **Verify (laptop):** `uv sync`; `uv run python -c "import gdp.detect"`; `uv run gdp detect --help`.

- [x] **2. Span→class mapping + box-format conversion (cheap sanity check, before any pipeline).**
      Build `PromptSpans.from_classes(classes, tokenizer=processor.tokenizer)`; runtime assertion
      that `decode_span(i) == classes[i]` for all 10 classes; positive_map masked-mean aggregation
      over query logits → `class_id`/`score`; cxcywh-normalized → xyxy-absolute conversion with
      `[0,1]` range guards.
      **Verify (laptop):** `uv run pytest tests/test_detect.py -q` — round-trip test asserts a box
      whose span decodes to "traffic light" is assigned class index 8, not 0.

- [x] **3. `gdp detect` end-to-end on fixture.** Wire batched inference through the detector +
      span→class mapping into the CLI command; write `runs/02-zeroshot/<ts>/predictions.json`
      (COCO-detection xywh format). Remove `detect` from `tests/test_cli.py`'s `PENDING` map; add
      fixture smoke assertions (exit 0, well-formed predictions file).
      **Verify (laptop):** `uv run gdp detect -c configs/default.yaml --dataset mini_bdd`;
      `uv run pytest tests/test_cli.py -q`.

- [x] **4. COCO mAP wrapper + operating-point P/R (unit-testable without the CLI).**
      `src/gdp/eval/coco_map.py` (pycocotools wrapper: mAP, mAP@50, per-class AP) and
      `src/gdp/eval/operating_point.py` (greedy IoU≥0.5 matcher → per-class + overall P/R,
      `sweep_threshold` helper).
      **Verify (laptop):** `uv run pytest tests/test_eval.py -q` — perfect-pred synthetic GT gives
      AP≈1.0; hand-built boxes give correct TP/FP/FN.

- [x] **5. `gdp evaluate` + metrics.json provenance.** `src/gdp/eval/metrics_io.py` assembling
      mAP/per-class AP/P-R/threshold/config echo/model_id/split/image count/git SHA/UTC
      timestamp/`is_synthetic`; wire the CLI command; remove `evaluate` from `PENDING`.
      **Verify (laptop):** `uv run gdp evaluate -c configs/default.yaml --dataset mini_bdd`; assert
      `metrics.json` has `is_synthetic: true` and the full provenance block; `uv run pytest
      tests/test_cli.py -q` green.

- [x] **6. Threshold sweep on train slice (leakage guard).** `sweep_threshold` plumbed into
      `gdp detect --split train --limit N`; `gdp evaluate` records the winning threshold and
      `"chosen_on": "train"`. On the fixture (val-only), sweep is skipped and the config default
      `box_threshold` is used and recorded as such.
      **Verify (laptop):** unit test on the sweep helper picks the expected threshold from a
      candidate list; assert the sweep never runs against `split=val`.

- [x] **7. SLURM job for full val (cluster, hand-off only).** Emit
      `scripts/slurm/zeroshot_eval.slurm` via the `slurm-job` skill: resumable, runs
      `gdp detect` + `gdp evaluate` with `-c configs/default.yaml -c configs/bdd100k.yaml
      --dataset bdd100k --split val`, writes `runs/02-zeroshot/<ts>/metrics.json`.
      **Verify (laptop):** script lints / dry-reads correctly (no in-session execution). Hand to
      user for `sbatch`.
