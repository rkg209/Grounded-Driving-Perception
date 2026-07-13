# Spec 03 — Fine-tune the detector (Stage-1 headline)

**Status:** draft · **Compute:** **cluster only** (SLURM) · **Depends on:** 01, 02

## Objective

Fine-tune Grounding-DINO on BDD100K train and report the **zero-shot → fine-tuned mAP delta** on
the same val split. This delta is the Stage-1 headline result and the project's core evidence that
domain adaptation of a foundation model works and is measurable.

## Inputs / outputs

- **In:** `IDEA-Research/grounding-dino-tiny`, BDD train (spec 01), the spec-02 baseline.
- **Out:** checkpoint (cluster, gitignored) + `runs/03-finetune/<ts>/metrics.json` reporting
  fine-tuned mAP **next to** the zero-shot mAP on the identical split, plus training curves (W&B).

## Approach

1. **Native HF training** — verified: `GroundingDinoForObjectDetection.forward` accepts
   `labels: list[dict]` with `class_labels` + `boxes` and computes the bipartite-matching loss
   (focal + L1 + GIoU). **No MMDetection dependency.**
   - `boxes` must be **normalized cxcywh** (the image processor's `do_convert_annotations` does
     this) — passing absolute xyxy trains a model that learns nothing. Assert this in the collator.
   - `class_labels` index the prompt's token spans (spec 01), not an arbitrary label list.
2. **Fixed prompt** of all 10 classes for every training image, so the contrastive head sees the
   full taxonomy each step.
3. Freeze the text encoder initially; fine-tune the vision backbone + fusion + heads. Ablate
   unfreezing the text encoder if time allows.
4. AdamW, lr ≈ 1e-4 (backbone ×0.1), cosine schedule, bf16, gradient checkpointing if memory-bound.
5. **Emitted as a SLURM script** via `/train-job` (CLAUDE.md §4). The agent never launches this
   in-session.
6. Overfit-a-tiny-subset sanity check **first**: train on ~20 images until loss → ~0. If that fails,
   the label mapping is wrong; no full run may start until it passes.

## Acceptance criteria

1. The 20-image overfit test drives loss to near-zero (proves labels reach the loss correctly).
2. A full run completes on the cluster and writes a checkpoint + `metrics.json`.
3. `metrics.json` contains **both** numbers on the **same** split: `map_zeroshot` (from spec 02) and
   `map_finetuned`, plus the delta, per-class AP, and full provenance.
4. Training curves logged (W&B), loss decreasing, no NaNs.
5. Qualitative "zero-shot miss → fine-tuned hit" pairs saved for the write-up.

## Honesty contract

- **H1** — fine-tuning a pretrained backbone. Never describe this as training a detector "from
  scratch", not even loosely.
- **H8** — the fine-tuned mAP is **meaningless without** the spec-02 baseline beside it on the same
  split. Both numbers ship together or neither ships.
- **H7** — if fine-tuning *hurts* a class (plausible for rare ones), report that class's regression.
  A mixed result honestly reported is stronger than a clean result quietly curated.

## Out of scope

No grounding phrases (04), no ONNX (05), no VLM. No hyperparameter search on val.

## Risks

- **Catastrophic forgetting of open-vocabulary ability**: fine-tuning on 10 fixed classes can
  degrade the model's ability to ground *novel* phrases — which is the whole selling point. Measure
  this: re-run a handful of out-of-taxonomy queries before/after and report the qualitative change.
  This is a known, expected tension and a strong interview talking point, not a failure to hide.
