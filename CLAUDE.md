# CLAUDE.md — Agent Constitution

**Project:** Grounded Driving Perception (Honda ADAS portfolio project)
**Charter (source of truth, do not edit):** `grounded-driving-perception-spec.md`
**Owner:** Rahul (IIT Bombay) — targeting Honda R&D Jobs 4 / 6 / 7.

You are the implementation agent for this project. Read this file before doing anything.

---

## 1 · What this project is

A two-stage vision-language driving-perception system, built spec by spec:

- **Stage 1 — Grounded detection.** Fine-tune `IDEA-Research/grounding-dino-tiny` (open-vocabulary detector) on BDD100K. Report zero-shot → fine-tuned **mAP** delta, plus **grounding accuracy** on a curated descriptive-phrase set. Then quantize, export to ONNX, and publish an accuracy-vs-latency curve.
- **Stage 2 — Driving-scene QA.** LoRA fine-tune **Qwen2.5-VL-3B** on **DriveLM**. Report base vs fine-tuned accuracy on the **official split with the official metric**, per-category, with failure analysis.

They are **two separate models**, evaluated separately. There is no joint model.

---

## 2 · The Honesty Register (H1–H8) — the highest law of this repo

This project's value to Honda is that **every claim is defensible**. Overclaiming destroys it. These
rules override convenience, override "it would look better if", and override my own suggestions.

| ID | Rule |
|----|------|
| **H1** | **No pretraining from scratch.** We adapt pretrained backbones and *measure* the domain gain. Say this proudly, never apologetically. |
| **H2** | Stage-2 VQA is scored **only** on DriveLM's official split with its official metric. Never self-invented questions, never a self-graded benchmark. |
| **H3** | **"Edge-deployed" means** quantized + ONNX-exported + latency-benchmarked on edge-class compute. It **never** means "deployed in a vehicle." |
| **H4** | **Camera-only.** No LiDAR, no radar, no sensor fusion. |
| **H5** | **No tracking, no trajectory prediction, no planning/control.** Single-frame perception + scene QA. The rest of the AD stack is downstream and out of scope. |
| **H6** | Stage 1 and Stage 2 are **separate models, evaluated separately.** Never imply one joint model. |
| **H7** | **Failures are measured and shown, not hidden.** Capability outside the evaluated set is a **demo, not a result**. Label it as such. |
| **H8** | **No metric claim without its paired baseline on the same split.** Stage-1 mAP requires the zero-shot number. Stage-2 accuracy requires the base-VLM number. |

**Applying the register:**
- Every spec declares which H-items it could violate ("Honesty contract").
- Before writing any README text, metrics table, or result claim, run `/claims-check`.
- If a task would require violating an H-item to succeed, **stop and tell the user** — do not quietly reinterpret the rule.
- A number without its baseline, its split, and its date is not a result. It is a rumour.

---

## 3 · The SDD loop — how work happens here

**Never write production code without an approved spec in `specs/`.**

```
/new-spec  →  /plan  →  /tasks  →  /implement (ONE task)  →  /verify  →  /progress
                ↑                                                          │
                └──────────────────  next task  ───────────────────────────┘
```

- `/implement` does **one task at a time**, then stops. No batching, no "while I'm here".
- Nothing is **done** until `/verify` passes **and** `progress_report.md` has an entry (§5).
- Spec status lives in `specs/README.md`. Only `approved` specs may be implemented.

---

## 4 · The laptop-vs-cluster rule (hard constraint)

This repo is developed on an **Apple M4, 16 GB, MPS**. It **cannot** fine-tune Grounding-DINO or a 3B VLM. Ever.

| Runs on the laptop | Runs on the IITB SLURM cluster |
|---|---|
| tests, linting, CLI, config | detector fine-tuning (spec 03) |
| synthetic-fixture pipelines | VLM LoRA fine-tuning (spec 07) |
| ONNX export + INT8 | full-split evaluation |
| **latency benchmarking** (the M4 *is* a legitimate edge target) | anything over a few hundred images |

**Never launch a full training run in-session.** Emit a SLURM script (`/train-job`) and hand it to the
user. A `guard-bash` hook enforces this; if it blocks you, that is the system working, not a bug.

Because datasets are large and not yet downloaded, **every spec must stay verifiable on
`tests/fixtures/` (a synthetic mini-BDD)**. If a feature can only be tested with the real dataset, it is
badly designed — add a fixture path.

---

## 5 · `progress_report.md` is MANDATORY

`progress_report.md` is the story of this project: what we did, **why**, **how**, what broke, and how we
fixed it. At the end it must be readable, alone, as the full development narrative.

**After every change that touches code, specs, configs, or results, append an entry.** The change is
**not complete** until the entry exists. Do not batch several changes into one entry. Do not rewrite or
delete past entries — the file is **append-only**, including the mistakes. The mistakes are the most
valuable part: they are what a Honda interviewer will ask about.

Entry template (also `/progress` and the `progress-log` skill):

```markdown
## [SEQ-000N] <short title>

**Date:** YYYY-MM-DD · **Spec:** <NN-name or "—"> · **Status:** <done | partial | reverted>

### What
<what changed, concretely — files, commands, behaviour>

### Why
<the reason. Tie to a spec, an H-item, or a bug. "Because it was next" is not a why.>

### How
<the approach, and the alternatives rejected + why>

### Issues & resolutions
<what broke and how it was fixed. Write "None." only if nothing broke.>

### Verification
<the exact commands run and their result. Not "tests pass" — show what passed.>
```

---

## 6 · Repo map

```
grounded-driving-perception-spec.md   the charter (immutable)
CLAUDE.md                             this file
progress_report.md                    append-only build narrative
specs/                                00–10 + README.md (status index)
src/gdp/                              the package (config, paths, seed, cli, …)
configs/                              YAML configs
tests/                                pytest + tests/fixtures/ (synthetic mini-BDD)
scripts/                              smoke.sh, slurm/ job templates
runs/                                 outputs, checkpoints, metrics (gitignored)
.claude/                              commands, skills, agents, hooks
```

---

## 7 · Conventions

- **Python ≥ 3.11**, managed with **uv**. `uv run <cmd>`, never bare `python`.
- **ruff** for lint+format. Type hints on public functions.
- Config is **YAML → dataclass**, validated on load. No magic constants in code.
- **Seed everything** (`gdp.seed.set_seed`). Device order: `cuda → mps → cpu`.
- Never commit `data/`, `runs/`, checkpoints, or `*.onnx` (see `.gitignore`).
- Results are written to `runs/<spec>/<timestamp>/metrics.json` — never pasted only into prose.
- Comments explain **why**, not what. Match surrounding style.

## 8 · Before you say "done"

1. `/verify` passed (real commands, real output — not assumed).
2. `progress_report.md` entry appended.
3. `/claims-check` clean if any user-facing text changed.
4. `specs/README.md` status updated.
