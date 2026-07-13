# Spec 04 — Curated grounding evaluation set

**Status:** draft · **Compute:** laptop (curation) + cluster (eval) · **Depends on:** 01, 03

## Objective

Detection scores *class names*. **Grounding** scores *descriptive phrases* — "the pedestrian near
the crosswalk", "the cyclist in the right lane". Build a small, honest, documented evaluation set of
such phrases over BDD frames and measure **grounding accuracy** (predicted box IoU ≥ 0.5 with the
referred object).

You **design and own this benchmark**, which is the point: evaluation design is a senior skill, and
an interviewer will probe how you avoided writing a benchmark that flatters your model.

## Inputs / outputs

- **In:** BDD val frames + their GT boxes (spec 01); the zero-shot and fine-tuned detectors.
- **Out:** `data/grounding_eval/phrases.json` — each entry: `image_id`, `phrase`, `target_box`,
  `qualifier_type`, `author`, `date`. Plus `runs/04-grounding/<ts>/metrics.json` with grounding
  accuracy for **both** models (H8).

## Approach

1. **Sample frames first, then write phrases** — never the reverse. Picking frames after seeing
   what the model gets right is how a benchmark becomes a advertisement.
2. Target ~150–300 phrases across qualifier types, reported **per type**:
   - **spatial** ("the car in the left lane", "the pedestrian on the far sidewalk")
   - **attribute** ("the white truck", "the person with an umbrella")
   - **relational** ("the cyclist behind the bus")
   - **negative/absent** ("the ambulance") on frames containing none — tests false-positive
     behaviour, which pure grounding sets usually ignore and which matters enormously for ADAS.
3. Each phrase must resolve to **exactly one** GT box, decided by the box's identity, not by what
   the model predicts. Ambiguous phrases are rewritten or dropped, with the drop count recorded.
4. Grounding accuracy = fraction of phrases whose top-1 predicted box has IoU ≥ 0.5 with the target.
   For negatives: correct = predicting nothing above threshold.
5. Document construction in the spec's own results section: who wrote the phrases, when, how
   ambiguity was resolved, how many were dropped and why.

## Acceptance criteria

1. `phrases.json` exists with ≥150 entries covering all four qualifier types.
2. A validation test: every phrase's `target_box` matches a real GT box in that image; no orphans.
3. Grounding accuracy reported for **both** zero-shot and fine-tuned models, **per qualifier type**.
4. The construction methodology is written down, including the drop count and the ambiguity rule.
5. Held-out discipline: no phrase was edited after seeing a model's prediction on it. State this.

## Honesty contract

- **H7** — this is a **small, self-built** evaluation set, not an official benchmark. Every mention
  of grounding accuracy must carry that caveat. It is a legitimate result *because* its construction
  is documented — not despite being self-built.
- **H8** — report both models. A grounding number for the fine-tuned model alone is unclaimable.
- **Bias risk (state it):** the author of the phrases also owns the model. The mitigations are
  (a) frames sampled before phrases were written, (b) phrases never edited after seeing predictions,
  (c) negatives included. Say this openly in the write-up; it is the honest answer to the obvious
  interview question.

## Out of scope

No crowd-sourcing, no RefCOCO-scale ambitions, no segmentation masks.
