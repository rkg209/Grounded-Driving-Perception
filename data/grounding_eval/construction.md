# Grounding evaluation set — construction record

**This is a small, self-built evaluation set, not an official benchmark.** It exists only because
CLAUDE.md §2 (H9) grants it a *named, sanctioned exception*: self-built benchmarks are otherwise
banned in this project, and this one is permitted **only** because its construction, bias, and drop
rate are fully documented here — and it is labelled self-built everywhere it appears (metrics.json's
`is_self_built_benchmark: true` / `caveat` fields, every README table, every write-up sentence).

This file is written **at authoring time**, alongside the phrases, not reconstructed afterward from
memory. It is the evidence a reviewer (or a Honda interviewer) checks against the mechanical guards
in `gdp.ground.phrases` and `gdp.ground.sample` — not a substitute for them.

---

## Who, when

- **Author:** Rahul (IIT Bombay), sole author of every phrase in `phrases.json`.
- **Date range:** _TODO — fill in when the real 150-300-phrase set is authored (post-BDD100K
  download). This template ships with spec 04's laptop-side implementation
  (`.claude/plans/04-grounding-eval-set.md` task 9); the real authoring pass is explicitly out of
  scope for that plan ("Not in this plan (manual, post-BDD)")._
- **Frame source:** BDD100K val split (`configs/bdd100k.yaml`), sampled via
  `gdp ground sample-frames --dataset bdd100k --n <grounding.num_frames> --seed <cfg.seed>`.

## The three bias mitigations (spec's own Honesty contract)

The author of the phrases also owns the model being scored — that conflict of interest is not
hidden, it is fenced in three specific, checkable ways:

1. **Frames are sampled before any phrase is written.** `gdp ground sample-frames` writes
   `frames.json` with a `sampled_at` ISO timestamp and the sampling `seed`, *before* the author has
   looked at what the model detects on those frames. A phrase set authored by first running the
   model and then writing prompts that happen to work would be exactly the "benchmark as
   advertisement" failure this record exists to rule out — `frames.json`'s timestamp is what makes
   the claim "sampling preceded authoring" checkable rather than merely asserted.
2. **`phrases.json` is hash-frozen and evaluation refuses to run against a changed file.**
   `gdp ground freeze` records `phrases.json`'s sha256 in `phrases.lock.json`; `gdp ground
   evaluate` calls `assert_frozen` and exits non-zero if the live file's hash no longer matches.
   Concretely: **no phrase was edited after seeing any model's prediction on it.** If a phrase later
   turns out ambiguous or wrong, it is fixed by *unfreezing, editing, and re-freezing before the
   next evaluation run* — never by editing it after inspecting a specific prediction.
3. **Negative phrases are mandatory, not optional.** Every qualifier-type coverage check in
   `gdp.ground.phrases.validate_phrases` requires at least one `negative` phrase; `min_phrases`
   (`grounding.min_phrases`, ≥150 for the real set) is spread across all four types by construction
   of the authoring process, not left to chance. A phrase set that only ever asks the model to find
   things that are really there cannot report a false-positive rate, and false positives are exactly
   what matters for an ADAS-relevant grounding claim.

## The ambiguity rule (verbatim)

> A phrase is valid only if exactly one GT box in the frame satisfies it. If two or more boxes
> plausibly satisfy the same phrase, the phrase is rewritten with a distinguishing qualifier (e.g.
> "the car" → "the car in the left lane"); if it cannot be made unique with a fair qualifier, it is
> **dropped**, and the drop is recorded below with a reason.

The mechanical half of this rule that `validate_phrases` can check without human judgment: any
`target_ann_id` with another GT box of the **same class** at **IoU ≥ 0.5** in the **same frame** is
flagged as `ambiguous` (a near-duplicate annotation — e.g. two heavily-overlapping `car` boxes,
which make "the car in the left lane" unresolvable by construction). The tool flags; the human
decides whether it's genuinely ambiguous or a legitimate close pair with a real distinguishing
qualifier already in the phrase text.

## Drop count and reasons

_TODO — fill in from `data/grounding_eval/drops.json` once the real phrase set is authored._ Every
drop must be recorded there as `{phrase, image_id, reason, date}`, and the count and rate must land
in the eventual `metrics.json`/write-up (H7: the drops are a file, not a memory). Until then:

| Reason | Count |
|---|---|
| ambiguous (near-duplicate same-class GT box) | _TODO_ |
| could not be made unique even after rewriting | _TODO_ |
| other (state explicitly) | _TODO_ |
| **Total dropped** | _TODO_ |
| **Total authored (kept + dropped)** | _TODO_ |
| **Drop rate** | _TODO_ |

## Held-out discipline (acceptance criterion 5)

**Stated explicitly, as the spec requires:** no phrase in the frozen set was edited after seeing a
model's prediction on it. This is enforced mechanically by the freeze/hash gate above (mitigation
2) — `phrases.lock.json`'s `frozen_at` timestamp and `sha256` are the checkable evidence, not just
this sentence.

## What this set is not

- Not RefCOCO-scale, not crowd-sourced, no segmentation masks (spec § Out of scope).
- Not a claim about grounding accuracy in general — only about this project's own detector(s),
  on this specific frozen phrase set, at this specific `box_threshold`, on this specific date.
- Not evidence on its own: every number computed against it must carry
  `is_self_built_benchmark: true` and this file's caveat, and must report **both** the zero-shot and
  fine-tuned model side by side (H8) — a fine-tuned-only grounding number is unpublishable.

## Fixture note (H7)

`tests/fixtures/grounding/phrases_mini.json` (8 phrases, `is_synthetic: true`) is a **separate,
committed, synthetic** phrase set over the 4-image `tests/fixtures/mini_bdd` fixture — it exists
only to keep every code path (`validate`, `freeze`, `sample-frames`, `evaluate`, `compare`)
verifiable offline (CLAUDE.md §4). It is not part of this construction record, is never reported as
a result, and is stamped `is_synthetic: true` everywhere it appears, exactly as `mini_bdd` itself is
(spec 00 §6).
