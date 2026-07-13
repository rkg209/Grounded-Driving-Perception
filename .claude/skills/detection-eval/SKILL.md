---
name: detection-eval
description: Correctly evaluate open-vocabulary detection — mAP, IoU, grounding accuracy, and the Grounding-DINO token-span trap. Use when computing, comparing, or reporting any detection or grounding metric (specs 02, 03, 04).
---

# Detection evaluation

A detection metric is easy to compute and easy to compute *wrongly* in a way that looks plausible.
These are the traps that actually bite in this project.

## 1. The token-span trap (the big one)

Grounding-DINO does **not** output class ids. It outputs, per box, a distribution over the **text
tokens** of the prompt. With the prompt `"pedestrian. rider. car. ..."`, a box "is a car" because it
scores highly on the tokens of the word *car*.

So evaluation needs `token span → class index`. Get this wrong and you get a model that trains fine,
runs fine, and reports near-zero mAP — or, worse, a plausible-but-wrong mAP.

- Build the mapping with the tokenizer's **offset mapping**, not by splitting on spaces. `traffic
  light` is two words and several word-pieces.
- **Round-trip test it:** decode the tokens in class *i*'s span and assert you get class *i*'s name.
  Do this for all 10 classes, especially the two-word ones.
- Symptom of a broken mapping: every prediction collapses onto class 0, or mAP ≈ 0 while the boxes
  visibly look correct. **If mAP is near zero, suspect the mapping before you suspect the model.**

## 2. Box formats — three of them, all in play

| Where | Format |
|---|---|
| COCO JSON / our annotations | absolute **xywh** |
| our `gdp.data.Box` | absolute **xyxy** |
| Grounding-DINO training labels | **normalized cxcywh** |

Converting in the wrong direction trains a model that learns nothing (loss plateaus immediately) or
evaluates a model that is actually fine as garbage. **Assert the format at every boundary.** A cheap
check: normalized coords must all lie in [0, 1]; if you see 640.0, you have a bug.

## 3. mAP

- Use a **pycocotools-compatible** evaluator. Do not hand-roll AP: the interpolation and the
  IoU-threshold averaging (0.50:0.95) have conventions, and a home-made number is not comparable to
  anything published.
- Report **mAP and mAP@50** and **per-class AP**. The aggregate hides the story — BDD100K is
  dominated by `car`, and a model can gain mAP while regressing on pedestrians. That regression is
  the safety-relevant one, and hiding it violates H7.
- **Always report the class-frequency table beside the AP table.** `train` is nearly absent in BDD;
  its AP is noise, and presenting it as a result is misleading.

## 4. Grounding accuracy (spec 04)

- Definition here: top-1 predicted box has **IoU ≥ 0.5** with the referred GT box.
- For **negative** phrases (the referred object is absent), correct = predicting *nothing* above
  threshold. Include these: a grounding model that confidently boxes a non-existent ambulance is
  exactly the ADAS failure mode worth measuring, and most grounding sets ignore it.
- Report **per qualifier type** (spatial / attribute / relational / negative). One aggregate number
  wastes the evaluation set you built.

## 5. Thresholds and leakage

- `box_threshold` / `text_threshold` change the metric. **Tune them on train, never on val.**
  Choosing the threshold that maximizes val mAP and then reporting that val mAP is leakage — the
  number is no longer an estimate of anything.
- Record the thresholds in `metrics.json`. A mAP without its threshold is unreproducible.

## 6. Reporting (H8)

Every detection number ships with:
- its **baseline on the same split** (zero-shot for spec 03),
- the split name and image count,
- the checkpoint id, the config, the thresholds, and the git SHA.

Write it to `runs/<spec>/<ts>/metrics.json` via `gdp.paths.run_dir`. **Never** report a metric
computed on `tests/fixtures/` — that is synthetic data we drew ourselves; it validates plumbing, not
perception (H7).
