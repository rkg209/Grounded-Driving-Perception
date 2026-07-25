# Grounded Driving Perception

Open-vocabulary road-user detection + language grounding + driving-scene VQA, edge-benchmarked.

> **Status: in development.** This README carries no results yet. Metrics tables appear here only
> when they have been produced by a completed spec, on a real split, next to their baseline.
> Until then there is nothing to claim. See `progress_report.md` for the build narrative.

## What it will do

Two stages, **two separate models, evaluated separately**:

1. **Grounded detection** — find any road user described in natural language ("the cyclist in the
   right lane"). Grounding-DINO fine-tuned on BDD100K. Reported as zero-shot → fine-tuned **mAP**
   (plus precision/recall), and **grounding accuracy** (IoU ≥ 0.5) on a curated descriptive-phrase set.
2. **Driving-scene QA, including risk reasoning** — answer questions about a scene ("is it safe to
   merge left?"). Qwen2.5-VL-3B LoRA fine-tuned on **DriveLM**, scored on the **official split with
   the official metric**, base vs fine-tuned, per category, with failure cases shown.

Then: quantize → ONNX → latency/FPS/memory benchmark → **accuracy-vs-latency curve**.

A demo composes both models into a readable **scene report**, with every line attributed to the model
that produced it.

## About "risk assessment"

Risk reasoning here is **measured, not asserted**: it is the accuracy on DriveLM's official
planning/behaviour/safety questions — real ground truth, real metric.

What this project deliberately does **not** do is ship a hand-written rule-based hazard engine with a
"risk detection accuracy" score. BDD100K is monocular with no calibration or depth ground truth, so
"distance to the lane boundary" isn't computable in metres, and scoring our own rules against our own
labels would be circular. Any heuristic risk overlay in the demo is badged as an unmeasured
illustration.

## Honest scope — what this is NOT

- **Not pretrained from scratch.** Pretrained backbones are adapted; the domain gain is measured.
- **Not deployed in a vehicle.** "Edge-deployed" means quantized, ONNX-exported, and
  latency-benchmarked on edge-class compute. Nothing here has been near a car.
- **The ONNX export is fixed-vocabulary, not open-vocabulary.** Exporting requires freezing the
  text prompt to the 10 BDD classes; the exported artifact answers only those classes, not
  arbitrary natural-language queries. The open-vocabulary path stays in PyTorch — a deliberate
  export tradeoff, stated in `metrics.json` as `fixed_prompt: true`, not a silent regression.
- **No LiDAR, radar, or sensor fusion.** Camera-only.
- **No tracking, no trajectory prediction, no planning or control.** Single-frame perception and
  scene understanding; the rest of the AD stack is downstream and deliberately out of scope.
- **Not one joint model.** Stage 1 and Stage 2 are separate models that share a theme, not weights.
- **The VLM can hallucinate.** Failures are reported per category, not hidden.
- **No self-invented benchmarks.** Capabilities without official ground truth — the scene report, any
  heuristic risk overlay, multi-frame reasoning — are shown as **labelled demos with no accuracy
  number**, never as results.

The full rules are the Honesty Register (H1–H9) in [`CLAUDE.md`](CLAUDE.md); they are enforced in
review by `/claims-check`.

## Quickstart

Runs offline on a laptop against a synthetic fixture — no dataset download, no GPU:

```bash
make install      # uv sync
make test         # pytest
make smoke        # end-to-end: import → config → device → fixture → CLI
uv run gdp info   # resolved config + device
uv run gdp --help # the full roadmap: implemented commands work, the rest name their spec
```

Real datasets (BDD100K, DriveLM) require registration and are prepared by spec 01 / spec 06.

## Layout

| Path | What |
|---|---|
| `grounded-driving-perception-spec.md` | The project charter (immutable) |
| `CLAUDE.md` | Agent constitution: honesty register + SDD loop |
| `specs/` | The spec backlog (00–10) and its status index |
| `progress_report.md` | Append-only story of the build: what, why, how, what broke |
| `src/gdp/` | The package |
| `tests/fixtures/mini_bdd/` | Synthetic scenes — **not real data**, never a source of results |

## Development

Spec-driven: `/new-spec → /plan → /tasks → /implement (one task) → /verify → /progress`.
No production code without an approved spec. Nothing is done until `progress_report.md` says how.
