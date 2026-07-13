# Spec 01 — BDD100K data pipeline

**Status:** draft · **Compute:** laptop (conversion) + cluster (storage) · **Depends on:** 00

## Objective

Turn raw BDD100K into the canonical COCO-style contract that spec 00's loader already speaks, and
build the **class-index → prompt-token-span mapping** that Grounding-DINO's contrastive text head
requires. This mapping is the single subtlest piece of Stage-1 data code; getting it wrong
produces a model that trains without error and detects nothing.

## Inputs / outputs

- **In:** BDD100K images (`100k/train`, `100k/val`) + detection labels (`det_20`). Requires
  registration at bdd-data.berkeley.edu — **not redistributable, never committed** (`.gitignore`).
- **Out:** `data/bdd100k/labels/det_20/det_{train,val}_coco.json` in the same schema as
  `tests/fixtures/mini_bdd/annotations.json`; loadable by the existing `gdp.data.load_dataset`
  with no changes to it.

## Approach

1. **Download + verify.** Document the exact registration steps and record file checksums. Data
   lives on the cluster; the laptop keeps only the fixture.
2. **Convert** BDD's native JSON → COCO xywh, using the 10-class taxonomy in
   `gdp.config.BDD100K_CLASSES` (order is load-bearing — it defines class indices).
3. **Prompt-span mapping** (`gdp/data/prompt.py`): the prompt is `"pedestrian. rider. car. ..."`.
   Grounding-DINO scores each query token against each box, so a class label must be expressed as
   the **token span** of that class name in the tokenized prompt. Build
   `class_idx → (tok_start, tok_end)` using the tokenizer's offset mapping, **never** by counting
   words — "traffic light" is two words and multiple word-pieces.
4. **Sanity report:** per-class box counts, image counts, degenerate/out-of-frame boxes dropped
   (with the count logged, not silently swallowed).
5. Extend `gdp data` CLI: `gdp data prepare --dataset bdd100k`.

## Acceptance criteria

1. `gdp data prepare` produces `det_val_coco.json` that `load_dataset` reads without modification.
2. A test asserts fixture-parity: the fixture and real BDD JSON pass the *same* validation.
3. Prompt-span test: for each of the 10 classes, decoding the tokens in its span returns the class
   name — including the multi-word `traffic light` / `traffic sign`.
4. Class-count report is written to `runs/01-data/<ts>/stats.json`; counts are non-zero for all 10.
5. Dropped-box count is reported explicitly.
6. All of the above are exercised on the fixture in CI-free laptop tests.

## Honesty contract

- **H7** — report the real class imbalance (BDD is dominated by `car`; `train` is near-absent).
  Do not quietly drop rare classes to flatter later mAP. If a class is too rare to evaluate, say so
  in the results table rather than omitting it.
- **H8** — the val split defined here is the split **all** later mAP numbers must use. Fixing it now
  is what makes zero-shot and fine-tuned comparable.

## Out of scope

No model, no inference, no training. No BDD segmentation/drivable-area labels (detection only).

## Risks

Registration may take days — this blocks 02/03 but **not** 04's design or 05's export work, both of
which can proceed on the fixture and on public COCO images.
