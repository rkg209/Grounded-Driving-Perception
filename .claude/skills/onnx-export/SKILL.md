---
name: onnx-export
description: Export Grounding-DINO to ONNX, quantize to INT8, and benchmark latency honestly (p50/p95/p99, warmup, accuracy-vs-latency curve). Use for spec 05 or any deployment/quantization/latency work.
---

# ONNX export, quantization, and honest latency

This is the "deployed" evidence for Honda J6/J7. It is also the easiest place in the project to
produce a number that is technically true and substantively misleading. Both halves matter.

## 1. The export blocker (know this before you start)

Grounding-DINO uses **custom multi-scale deformable-attention kernels**, which are **not
ONNX-exportable**. The export will fail until you set:

```python
config.disable_custom_kernels = True   # swaps in the pure-PyTorch attention path
```

This field already exists in `gdp.config.DetectorConfig` and is documented in the HF
`GroundingDinoConfig` docs as *"necessary for the ONNX export"*. Keep it `False` for training
(faster) and `True` for export.

## 2. The text-input problem — and the honest way to state it

Grounding-DINO takes **image + text**. The text branch has dynamic sequence length, which makes a
fully general open-vocabulary ONNX graph painful.

The pragmatic export: **freeze the prompt** to the fixed 10-class string, so the text tokens are a
constant and the exported graph is image-in / boxes-out.

That is a legitimate engineering choice — deployed detectors run a fixed taxonomy — **but it means
the ONNX artifact is not open-vocabulary.** Say that, out loud, in the README and the spec. Shipping
a fixed-class ONNX model while the README sells "open-vocabulary detection, deployed" is precisely
the kind of quiet overclaim H3/H7 exist to prevent. The honest framing is strong on its own:
*"the open-vocabulary model is fine-tuned and evaluated; the deployed artifact fixes the taxonomy to
the 10 driving classes, and exporting the text encoder separately is the documented next step."*

## 3. Quantization

1. **Dynamic INT8** first (`onnxruntime.quantization.quantize_dynamic`) — no calibration data, works
   well for the transformer/linear layers that dominate this model.
2. **Static INT8** if dynamic underperforms — needs a calibration set drawn from **train**, never val.
3. Then **re-measure mAP**. Quantization is not free, and the whole deliverable is the *tradeoff*.

## 4. Benchmarking latency honestly

The methodology is the result. Do it properly:

- **Warm up ≥50 iterations and discard them.** The first inferences include lazy init, memory
  allocation and thermal ramp; including them inflates latency and makes your speedup look better
  than it is.
- Then **≥200 timed iterations**.
- Report **p50, p95, p99, and FPS**. **Never report only the mean.** A mean hides the tail, and in a
  perception loop the tail is what drops frames. p99 is the number an ADAS engineer actually asks for.
- **Fix and record**: thread count, batch size, input resolution, hardware (Apple M4, 16 GB), ONNX
  Runtime version. A latency number without its hardware is meaningless.
- Benchmark **the same input resolution** as the accuracy evaluation. Speeding up by silently
  shrinking the input and then quoting the fp32 mAP is a lie with extra steps.
- Beware thermal throttling on a laptop: run variants **interleaved**, not one after the other, or
  the last one measured looks worst. Report that you did.

## 5. The deliverable

`curve.png` — **accuracy (mAP) vs p50 latency**, one point per variant (fp32 PyTorch, fp32 ONNX,
INT8 ONNX). Plus `latency.json` and `metrics.json` with all of the above.

A speed number without its accuracy number violates **H8** and is worth nothing to a reviewer: of
course it got faster, you made it worse. The question is *how much worse, for how much faster* —
that is the engineering, and that is the plot.

## 6. Language discipline (H3)

"Edge-deployed" here means: **quantized + ONNX-exported + latency-benchmarked on edge-class
compute.** It does **not** mean installed in a vehicle. Never write "in-car", "production", or
"real-time in a vehicle". No TensorRT/Jetson claim unless a Jetson was actually used — if not, it is
future work, and saying so costs nothing.
