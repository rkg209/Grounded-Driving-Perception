---
description: Audit every claim in the repo against the Honesty Register (H1–H8)
argument-hint: [file or "all"]
---

Audit **$ARGUMENTS** (default: `README.md`, `specs/`, `progress_report.md`, and any results tables)
against the Honesty Register in `CLAUDE.md` §2.

Use the `claims-auditor` sub-agent for a full-repo pass; do it inline for a single file.

## What to look for

For every claim, ask: **"If a Honda interviewer pushed on this sentence, would it survive?"**

| Check | The violation looks like |
|---|---|
| **H1** | "trained a detector", "built a VLM", "trained from scratch" — when we fine-tuned a pretrained backbone |
| **H2** | a VQA number from self-written questions, a re-split, or a self-graded metric |
| **H3** | "deployed", "in-car", "production", "real-time in a vehicle" — without the quantized/ONNX/benchmarked definition attached |
| **H4** | any mention of LiDAR, radar, or fusion as something we did |
| **H5** | "predicts what the pedestrian will do", "tracks", "plans" — we do none of these |
| **H6** | "the system detects objects and answers questions" phrased as one model; a demo implying Stage 1 feeds Stage 2 |
| **H7** | a capability shown only qualitatively but described as a result; a metrics table with the bad class quietly missing; a metric computed on the synthetic fixture |
| **H8** | a headline number with no baseline on the same split; a latency win with no accuracy number |

Also flag: numbers with no `runs/*/metrics.json` behind them, superlatives ("state-of-the-art",
"novel"), and any table that was hand-typed rather than generated.

## Output

For each finding: the **file and line**, the **offending sentence**, **which H-item** it breaks, and
a **rewrite that is both honest and strong**. Honesty here is not self-deprecation — "we adapt
pretrained foundation models and measure the domain gain" is a *more* impressive sentence than
"we trained a detector", and it is also true.

If nothing is wrong, say so plainly. Do not invent findings to look thorough.
