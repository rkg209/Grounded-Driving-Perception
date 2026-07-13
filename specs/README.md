# Spec backlog

The project is built one spec at a time. **Only `approved` specs may be implemented.** Update the
status column here as part of the change that moves a spec (CLAUDE.md §8).

| # | Spec | Status | Compute | Headline output |
|---|------|--------|---------|-----------------|
| 00 | [scaffold](00-scaffold.md) | **done** | laptop | package, config, CLI, fixtures, tests, smoke |
| 01 | [data-bdd100k](01-data-bdd100k.md) | draft | laptop + cluster | BDD100K → COCO-style; class↔prompt-span mapping |
| 02 | [zeroshot-baseline](02-zeroshot-baseline.md) | draft | cluster (laptop subset) | **zero-shot mAP — the bar for H8** |
| 03 | [finetune-detector](03-finetune-detector.md) | draft | cluster | **zero-shot → fine-tuned mAP delta (Stage-1 headline)** |
| 04 | [grounding-eval-set](04-grounding-eval-set.md) | draft | laptop + cluster | curated phrase set + grounding accuracy |
| 05 | [deploy-onnx-edge](05-deploy-onnx-edge.md) | draft | laptop | INT8 + ONNX + **accuracy-vs-latency curve** |
| — | **◄ CHECKPOINT** | | | *project is complete and defensible here* |
| 06 | [data-drivelm](06-data-drivelm.md) | draft | laptop + cluster | DriveLM official splits, train subset |
| 07 | [finetune-vlm](07-finetune-vlm.md) | draft | cluster | Qwen2.5-VL-3B LoRA on DriveLM |
| 08 | [vqa-eval-failure-analysis](08-vqa-eval-failure-analysis.md) | draft | cluster | **base vs fine-tuned VQA accuracy (Stage-2 headline)** + **risk reasoning (planning/behaviour categories)** + failures |
| 09 | [integrated-demo](09-integrated-demo.md) | draft | laptop | Gradio: query→boxes, question→answer, **attributed scene report** |
| 10 | [writeup](10-writeup.md) | draft | laptop | README, metrics tables, failure analysis, repro |
| 11 | [temporal-demo](11-temporal-demo.md) | draft **(stretch)** | laptop | multi-frame VLM reasoning — **demo only, no metric (H9)**. First to cut. |

## The checkpoint rule

Spec 05 is the line the charter draws. If the schedule slips, **scope degrades, viability does
not**: after 05 the project is already "a fine-tuned, grounded, edge-deployed driving detector with
measured gains". Specs 06–08 (the VQA stage) are additive. Never sacrifice 00–05 to start 07 early.

**Cut order, if time runs out:** 11 first, then 09's extras, then 06–08 as a block. Never 00–05.

## Where "risk assessment" lives

There is no separate risk-engine spec, and that is deliberate. **DriveLM's official
planning/behaviour/safety categories *are* the risk benchmark** — official questions, official ground
truth, official metric — so risk reasoning is *measured* in **spec 08** rather than asserted by
hand-written rules.

A rule-based hazard engine was considered and rejected: BDD100K is monocular with no calibration or
depth ground truth, so "distance to the lane boundary" is not computable in metres, and scoring our
own rules against our own labels would be a self-invented benchmark (**H9**). A heuristic risk overlay
may still appear in the **demo (spec 09)**, badged as unmeasured, with no score attached.

## Anatomy of a spec

Every spec carries these sections. The two that make it a *spec* rather than a wish:

- **Acceptance criteria** — falsifiable. "Works well" is not a criterion; "mAP is written to
  `runs/<spec>/*/metrics.json` alongside the zero-shot baseline on the same split" is.
- **Honesty contract** — which H-items (CLAUDE.md §2) this spec could violate, and the specific
  thing that would violate them. This is what `/claims-check` audits against.

## Status vocabulary

`draft` → written, not yet reviewed · `approved` → human-reviewed, may be implemented ·
`in-progress` → being implemented, one task at a time · `done` → acceptance criteria met, verified,
and narrated in `progress_report.md`.
