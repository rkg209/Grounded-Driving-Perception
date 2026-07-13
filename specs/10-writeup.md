# Spec 10 — Write-up

**Status:** draft · **Compute:** laptop · **Depends on:** 02, 03, 04, 05, 08, 09

## Objective

The public artifact: a README that a Honda reviewer can read in five minutes and trust — architecture
diagram, every metrics table **with its baseline**, the failure analysis, the honest-scope section,
and reproduction instructions that actually work.

## Inputs / outputs

- **In:** every `runs/*/metrics.json`, the failure analysis (08), the demo video (09).
- **Out:** the final `README.md`, `docs/architecture.png`, a public repo.

## Approach

1. **Metrics tables are generated from `runs/*/metrics.json`, never hand-typed.** Hand-typing a
   number is how a number drifts from what the code produced. A small script renders them.
2. Required tables, each with **both** columns (H8):
   - Stage 1: zero-shot mAP **vs** fine-tuned mAP, overall + per class.
   - Grounding: zero-shot **vs** fine-tuned accuracy, per qualifier type (04).
   - Deployment: accuracy-vs-latency across fp32 / ONNX / INT8, with p50/p95/p99 + FPS (05).
   - Stage 2: base **vs** fine-tuned VQA on the official DriveLM val split, per category (08).
3. An **Honest scope** section, verbatim from the register: no pretraining from scratch; not
   deployed in a vehicle; camera-only; no tracking/prediction/planning; two separate models; the
   grounding set is self-built; the VLM hallucinates at a measured rate.
4. A **Limitations & next steps** section: single-frame, camera-only, self-built grounding set, no
   Jetson/TensorRT, VLM trained on a subset. Each limitation gets its concrete next step.
5. **Reproduction:** exact commands, exact configs, exact SLURM scripts, dataset registration links,
   expected runtimes. Then *actually follow them from a clean clone* — an unreproduced repro section
   is a claim like any other.
6. The interview one-liners from the charter (Appendix) get a section — they are the artifact's
   defence in conversation.
7. **Final `/claims-check` over the whole repo** against H1–H8. Nothing ships until it is clean.

## Acceptance criteria

1. Every number in the README traces to a `metrics.json` (a test asserts the tables were generated,
   not typed).
2. Every headline metric appears beside its baseline.
3. Architecture diagram shows Stage 1 and Stage 2 as **separate** models.
4. Honest-scope and Limitations sections present.
5. Repro instructions executed from a clean clone at least once; deviations fixed.
6. `/claims-check` clean; `claims-auditor` finds no H-violations.
7. `progress_report.md` reads, end to end, as a coherent story of the build.

## Honesty contract

**All of H1–H8.** This is the spec where overclaiming would actually reach a reader, so it is where
the register earns its keep. The reflex to round a number up, to drop the losing class from a table,
or to write "deployed" without its definition — that reflex shows up here. Do not indulge it: the
project's entire differentiator is that its claims survive scrutiny.

## Out of scope

No paper submission, no blog-post marketing language, no benchmark leaderboard claims.
