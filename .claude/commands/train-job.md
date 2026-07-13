---
description: Generate a SLURM script for a training run (never train in-session)
argument-hint: <NN-spec-name> [notes]
---

Generate a SLURM job script for **specs/$ARGUMENTS.md**.

The laptop cannot fine-tune Grounding-DINO or a 3B VLM (CLAUDE.md §4). Training happens on the IITB
cluster, launched **by the user**, never by the agent in-session.

## Before writing the script

Confirm the cheap sanity checks already passed — a bad label mapping wastes hours of GPU time and is
free to catch on the laptop:

- **Spec 03:** the 20-image overfit test drove loss → ~0, and boxes are normalized cxcywh.
- **Spec 07:** the 16-sample overfit test drove loss → ~0, and the label mask is answer-tokens-only.

If they have not passed, **say so and stop.** Do not queue a job on an unvalidated pipeline.

## The script (`scripts/slurm/<spec>.sbatch`)

Include: partition/GPU/time/memory requests, module or `uv` environment setup, `WANDB_*` env,
a **resumable** checkpoint path under `runs/<spec>/`, stdout/stderr to `runs/<spec>/slurm-%j.out`,
and the exact `uv run gdp ...` command with its configs.

Make it **resumable**: cluster jobs get pre-empted, and losing eight hours of training to a walltime
kill is a self-inflicted wound.

## Then

1. Print the script and the exact `sbatch` command for the user to run.
2. Tell them what to watch (`squeue`, the W&B run, the first loss values — and what "wrong" looks
   like: loss flat at the start means the labels aren't reaching the loss).
3. **Do not run it.** Hand it over.
4. Append a `progress_report.md` entry recording the job that was prepared and why.
