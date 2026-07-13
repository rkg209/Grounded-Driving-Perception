# Spec 08 — VQA evaluation + failure analysis (Stage-2 headline)

**Status:** draft · **Compute:** cluster · **Depends on:** 06, 07

## Objective

Evaluate **base vs LoRA-fine-tuned** Qwen2.5-VL on DriveLM's **official val split** with its
**official metric**, break the result down **per question category**, and analyse the failures —
including hallucinations — openly.

## This spec *is* the risk-assessment result

DriveLM's official taxonomy already contains **prediction / planning / behaviour** questions —
"is it safe to merge left?", "what should the ego vehicle watch for?". Those questions *are* risk
reasoning, and they come with **official ground truth and an official metric**.

So risk assessment is delivered here, measured, rather than as a separate hand-rolled engine. This is
deliberate and it is the honest path:

- A rule-based risk engine (distance thresholds, relative position) would need **metric distance**,
  which a monocular BDD100K frame **cannot give** — no calibration, no depth ground truth. Pixel
  proximity is not distance, and dressing it up as one is pseudo-precision.
- "Risk detection accuracy" against labels we invented, scored by rules we wrote, is **not evidence**
  (H9). It is the unfalsifiable demo the charter explicitly warns against.

**Report the planning/behaviour categories as the risk headline**, and say plainly that risk here
means *"accuracy on the benchmark's safety-reasoning questions"*, not a calibrated hazard probability.
A heuristic risk overlay may appear in the demo (spec 09) as an explicitly **unmeasured** illustration.

## Inputs / outputs

- **In:** base `Qwen2.5-VL-3B-Instruct`, the spec-07 adapter, `data/drivelm/val.jsonl` (official).
- **Out:** `runs/08-vqa/<ts>/metrics.json` (both models, overall + per category),
  `predictions.jsonl` (every prediction, so any number can be audited), `failures.md`.

## Approach

1. **Use DriveLM's official evaluation code and metric.** Do not reimplement the scorer "for
   convenience" — a home-made scorer is a home-made result. If the official metric composes several
   sub-metrics (accuracy, language scores, GPT-score), report the composition, and if a sub-metric
   requires an API judge we cannot afford, **report the sub-metrics we can compute and say plainly
   which we omitted and why**.
2. Evaluate the **base model first**, then the fine-tuned one, on the identical split, with
   identical generation settings (greedy; record `max_new_tokens`, temperature).
3. **Per-category accuracy** (perception / prediction / planning / behaviour). A single aggregate
   number hides the story: models typically gain most on perception and least on planning. The
   planning/behaviour rows are the **risk-reasoning headline** — and if the fine-tune gains on
   perception while planning stays flat, that is the honest and interesting finding, not something
   to average away.
4. **Failure analysis** (`failures.md`) — sample ≥30 errors, categorise them, and show images:
   - hallucinated objects that are not in the scene,
   - correct object, wrong attribute/count,
   - plausible-but-ungrounded reasoning (the dangerous one for ADAS),
   - format/parse failures.
5. Report where fine-tuning **hurt**, if it did.

## Acceptance criteria

1. `metrics.json` contains base **and** fine-tuned scores on the full official val split, overall
   and per category, with generation settings + provenance.
2. The metric is DriveLM's official one; any omitted sub-metric is named and justified in the JSON.
3. `predictions.jsonl` is complete — every val item, so the headline number is reproducible from raw
   outputs.
4. `failures.md` contains ≥30 categorised failures with images and honest commentary.
5. Hallucination rate is quantified, not merely mentioned.

## Honesty contract

- **H2** — official split, official metric. No self-invented questions. No self-grading.
- **H9** — **no invented risk metric.** "Risk detection accuracy" must never appear unless it is
  DriveLM's own scored categories. A hazard score from our own rules gets **no number**, ever.
- **H5** — answering DriveLM's planning questions in language is in scope; **shipping a planner is
  not.** We output an answer to a benchmark question, never a driving command.
- **H8** — the fine-tuned number ships **only** beside the base number. The delta is the result.
- **H7** — **failures are a deliverable, not an embarrassment.** "Does your VLM hallucinate?" →
  "Yes; here is the rate, the categories, and the examples." That answer is worth more to a Honda
  interviewer than a suspiciously clean score.
- **H6** — Stage-2 accuracy says nothing about Stage-1 mAP. Never merge them into one "system
  accuracy".

## Out of scope

No cherry-picked demo reel in place of metrics. No comparison to GPT-4V-class models we cannot run
under identical conditions (an unfair comparison is not a result).
