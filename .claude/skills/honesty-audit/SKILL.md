---
name: honesty-audit
description: Apply the Honesty Register (H1–H9) to any claim, README text, metrics table, spec, or commit message before it ships. Use whenever writing user-facing text or reporting a number.
---

# Honesty audit

This project's entire differentiator is that **every claim survives scrutiny**. A Honda reviewer who
catches one overclaim will discount everything else in the repo — and they will be right to. The
register is not modesty; it is the source of the project's credibility.

## The register (CLAUDE.md §2)

| ID | Rule | The tempting violation |
|----|------|------------------------|
| H1 | No pretraining from scratch | "trained a detector", "built a VLM" |
| H2 | Official DriveLM split + official metric only | self-written questions; re-splitting; self-grading |
| H3 | "Edge-deployed" = quantized + ONNX + benchmarked | "deployed", "in-car", "production", "real-time in a vehicle" |
| H4 | Camera-only | any LiDAR/radar/fusion phrasing |
| H5 | No tracker/predictor/planner **as a module** (answering DriveLM's planning *questions* is fine) | "predicts pedestrian intent", "plans a manoeuvre", any driving command |
| H6 | Two separate models | "the system sees and reasons" as one pipeline; an unattributed composed scene report |
| H7 | Failures measured, not hidden | a table missing its worst class; a demo shown as a result; a metric from the synthetic fixture |
| H8 | No metric without its paired baseline | fine-tuned mAP alone; a latency win with no accuracy |
| H9 | **No self-invented benchmark or metric** | "risk detection accuracy: 87%" from our own rules and our own labels; a temporal-QA score on questions we wrote; **any** number attached to the scene report or risk overlay |

## The test to apply to every sentence

> **If a Honda interviewer read this sentence aloud and asked "show me", would it survive?**

If the honest version needs a caveat, **the caveat goes in the sentence**, not in a footnote nobody
reads.

## Honest ≠ weak

This is the part people get wrong. The honest phrasing is usually the *more* impressive one, because
it demonstrates that you know what you actually did:

| Weak-and-false | Strong-and-true |
|---|---|
| "Trained an open-vocabulary detector for driving." | "Adapted a pretrained open-vocabulary detector to driving data and **measured** the gain: zero-shot X → fine-tuned Y mAP on BDD100K val." |
| "Deployed on edge devices." | "Quantized to INT8, exported to ONNX, and benchmarked on Apple M4: p50 A ms, p99 B ms, C FPS, at a D-point mAP cost — here is the accuracy-vs-latency curve." |
| "Built a VLM that understands driving scenes." | "LoRA fine-tuned Qwen2.5-VL-3B on DriveLM; official-split accuracy went from X (base) to Y, with per-category breakdown and a measured hallucination rate." |
| "Achieves high accuracy." | "mAP 0.41 vs a 0.29 zero-shot baseline on the same split; pedestrian AP regressed by 0.02, which we report and explain." |

The right column is what a strong candidate sounds like. The left column is what everyone else's repo
says.

## The "risk" trap specifically

"Risk assessment" is the most seductive overclaim available to this project, because it sounds like
exactly what an ADAS team wants. The rule:

- **Measured risk = DriveLM's official planning/behaviour/safety categories** (spec 08). This is a
  real, defensible result.
- **A hazard score from our own heuristics gets NO number** (H9). BDD100K has no depth or calibration,
  so pixel proximity is not distance; and "risk detection accuracy" against labels we invented,
  scored by rules we wrote, is circular. It ships as a badged demo or not at all.

If you see `Risk Level: HIGH` next to anything resembling a percentage, that is an H9 violation.

## Specific things to catch

- **A number with no baseline** — the most common violation. Half a comparison is not a result.
- **A metric computed on `tests/fixtures/`** — that data is synthetic and we drew it. It tests
  plumbing. It is *never* a result.
- **A missing class** in a per-class table. If it regressed, that is the interesting part.
- **Hand-typed numbers** that do not match `runs/*/metrics.json`. Generate tables; never type them.
- **Superlatives**: "state-of-the-art", "novel", "production-grade". Delete them; they add nothing
  and invite a challenge you cannot win.
- **The word "deployed"** anywhere without its definition attached.

## When a result is disappointing

Report it. A fine-tune that gains little, an INT8 model that loses real accuracy, a VLM that
hallucinates often — **these are findings**, and analysing them honestly is a stronger signal than a
clean number nobody can reproduce. The failure analysis is a deliverable (spec 08), not a confession.
