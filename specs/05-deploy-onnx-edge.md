# Spec 05 — Quantize, export, benchmark ◄ **CHECKPOINT**

**Status:** draft · **Compute:** **laptop** (the M4 *is* the edge target) · **Depends on:** 03

## Objective

Prove deployability: quantize the fine-tuned detector, export it to ONNX, and publish the
**accuracy-vs-latency curve** on edge-class compute. This is the literal wording of Honda J6
("edge device deployment of machine learning models") and J7 ("training and **deploying**
vision-based AI models").

> **This spec is the checkpoint.** When it lands, the project is complete and defensible on its own:
> *a fine-tuned, language-grounded, edge-benchmarked driving detector with measured gains.*
> Everything after (06–08, the VQA stage) is additive. Schedule overrun degrades scope, not viability.

## Inputs / outputs

- **In:** the spec-03 fine-tuned checkpoint; BDD val (for the accuracy half of the curve).
- **Out:** `runs/05-deploy/<ts>/` — `model.onnx`, `model.int8.onnx`, `latency.json`, `curve.png`,
  `metrics.json`.

## Approach

1. **Export precondition (verified against HF docs):** set
   `GroundingDinoConfig.disable_custom_kernels = True` before export. Grounding-DINO's custom
   multi-scale deformable-attention CUDA/CPU kernels **are not ONNX-exportable**; the config flag
   swaps in the pure-PyTorch path. Without this, export fails — this is the known blocker and the
   config field already exists in `gdp.config.DetectorConfig`.
2. **Export** with `torch.onnx.export` (dynamo path preferred), dynamic axes for batch. The text
   branch has dynamic sequence length; **fix the prompt to the 10-class string and freeze the text
   tokens** so the exported graph has a static text input — the detector's deployed contract is
   "10 driving classes", and the open-vocabulary path stays in PyTorch. **State this limitation
   plainly**: the ONNX artifact is not open-vocabulary. (Alternative — export the text encoder
   separately and cache embeddings — is the documented next step.)
3. **Quantize:** ONNX Runtime dynamic INT8 first (simplest, no calibration); static INT8 with a
   calibration set from train if dynamic disappoints.
4. **Benchmark** on the M4 CPU via ONNX Runtime: ≥50 warmup iterations (discard), then ≥200 timed;
   report **p50, p95, p99, and FPS** — never a bare mean, which hides tail latency that matters for
   a real perception loop. Fix threads and state them.
   Also record **memory**: on-disk model size (fp32 vs INT8) and **peak RSS during inference**. An
   edge reviewer's second question after latency is always "does it fit?" — a model that hits 30 FPS
   but needs 6 GB of RAM is not deployable, and reporting speed without footprint tells half the
   story.
5. **Re-evaluate mAP** for each variant (fp32 PyTorch, fp32 ONNX, INT8 ONNX) on the same val split,
   then plot accuracy vs p50 latency. That plot *is* the deliverable.

## Acceptance criteria

1. `model.onnx` loads in ONNX Runtime and its outputs match PyTorch within tolerance on fixture
   images (a numerical parity test, not a vibe check).
2. INT8 model produced and benchmarked.
3. `latency.json`: p50/p95/p99 + FPS **+ model size on disk + peak RSS** for each variant, with
   hardware, thread count, and iteration count recorded.
4. `metrics.json`: mAP for **all three** variants on the same val split — quantifying the accuracy
   *cost* of quantization, not just its speed benefit.
5. `curve.png`: accuracy vs latency, one point per variant.
6. The ONNX artifact's fixed-prompt limitation is stated in the spec's results and in the README.

## Honesty contract

- **H3** — **"edge-deployed" = quantized + exported + latency-benchmarked on edge-class compute.**
  The words "deployed in a vehicle", "in-car", or "production" must never appear. The M4 laptop is
  an honest edge-class target; a Jetson would be better and is stated as future work, not claimed.
- **H8** — a latency win with no accuracy number beside it is a half-truth. INT8 mAP must be
  reported even if it drops. **Especially** if it drops: the accuracy-vs-latency *tradeoff* is the
  result, not "it got faster".
- **H7** — if INT8 accuracy collapses, report it and the curve showing it. That is a finding.

## Out of scope

- **TensorRT** — no Jetson available. It is named as future work, never as something we did.
- **OpenVINO** — optional and Intel-only; the M4 is our edge target, so it is not attempted. If it is
  ever run, it is an *additional* point on the curve, not a replacement for the honest one.
- **No VLM export** — a 3B VLM to ONNX is a project of its own.
