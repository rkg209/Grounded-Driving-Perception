# Spec 06 — DriveLM data pipeline (Stage 2 begins)

**Status:** draft · **Compute:** laptop (conversion) + cluster (storage) · **Depends on:** 00

## Objective

Prepare **DriveLM** (nuScenes-based driving VQA) into a chat-formatted training set for Qwen2.5-VL,
while preserving its **official train/val split** exactly. Everything Stage 2 claims rests on that
split being untouched (H2).

## Inputs / outputs

- **In:** DriveLM annotations + nuScenes camera images (both require registration; not
  redistributable, never committed).
- **Out:** `data/drivelm/{train,val}.jsonl` — one record per QA pair: `image_paths`, `question`,
  `answer`, `category`, `scene_token`, `official_split`.

## Approach

1. **Download + register.** nuScenes and DriveLM are separate registrations; document both.
2. **Preserve the official split verbatim.** Do not re-shuffle, do not re-split, do not "rebalance".
   If a scene is in official val, it is in our val — full stop.
3. **Scene-level leakage check.** nuScenes frames within a scene are near-duplicates (2 Hz from a
   continuous drive). A frame-level split would leak val into train and inflate accuracy massively.
   Assert **zero `scene_token` overlap** between our train and val. This test is non-negotiable.
4. **Chat formatting** for Qwen2.5-VL: `<image>` + question → answer, using the model's chat
   template (not a hand-rolled prompt string).
5. **Tractable train subset** (cluster budget): subsample **by scene**, never by frame, keeping the
   category distribution. Record exactly what was sampled and the sampling seed.
6. **Preserve question categories** (perception / prediction / planning / behaviour) — spec 08
   reports per-category accuracy, which is impossible if the category is dropped here.
7. A synthetic mini-DriveLM fixture (like spec 00's mini-BDD) so Stage-2 code is laptop-testable.

## Acceptance criteria

1. `train.jsonl` / `val.jsonl` produced; val is **exactly** the official val split.
2. Scene-overlap test passes: `set(train.scene_token) ∩ set(val.scene_token) == ∅`.
3. Category distribution reported for both splits.
4. The train subsample is documented (size, seed, sampling unit = scene) and reproducible.
5. A synthetic VQA fixture exists and Stage-2 code paths run against it offline.
6. Chat formatting round-trips through the real Qwen2.5-VL processor.

## Honesty contract

- **H2** — the **official split and official metric, or nothing.** Never invent questions, never
  re-split, never evaluate on train. Subsampling is permitted **for training only** and must be
  disclosed ("fine-tuned on N% of DriveLM train, evaluated on the full official val").
- **H7** — if compute forces a small train subset, that is a stated limitation of the result, not a
  detail to bury.

## Out of scope

No LiDAR from nuScenes (**H4**: camera-only). No multi-frame/temporal QA (**H5**). No new questions.
