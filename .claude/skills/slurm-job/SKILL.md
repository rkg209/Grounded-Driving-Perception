---
name: slurm-job
description: Write resumable SLURM job scripts for cluster training (specs 03, 07). Use whenever a training run is needed — training never runs in-session on the laptop.
---

# SLURM jobs

The M4 laptop cannot fine-tune Grounding-DINO or a 3B VLM. Training runs on the IITB cluster,
**launched by the user**. The agent writes the script and hands it over (CLAUDE.md §4).

## Before you write the script — earn the GPU hours

A bad label mapping costs hours of cluster time and produces a model that is silently useless. Every
one of these is free to check on the laptop, and each one has burned somebody before:

- **Overfit a tiny subset first.** 20 images (spec 03) or 16 QA pairs (spec 07) → loss must go to
  ~0. If it does not, the labels are not reaching the loss. **No full run may be queued until this
  passes.** This single check catches most of what follows.
- **Spec 03:** boxes are normalized **cxcywh**; `class_labels` index prompt **token spans**.
- **Spec 07:** loss is masked to **answer tokens only** (question + image tokens are `-100`).
- Fix and record the seed.

**Loss that is flat from step 1 is not a learning-rate problem. It is a labels problem.**

## Script skeleton (`scripts/slurm/<spec>.sbatch`)

```bash
#!/bin/bash
#SBATCH --job-name=gdp-<spec>
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=runs/<spec>/slurm-%j.out
#SBATCH --error=runs/<spec>/slurm-%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export WANDB_PROJECT=grounded-driving-perception
export HF_HOME="$SCRATCH/hf"          # never fill the home quota with weights
export TOKENIZERS_PARALLELISM=false

uv run gdp train \
  -c configs/default.yaml \
  -c configs/<spec>.yaml \
  --resume-from-latest              # cluster jobs get pre-empted; always resumable
```

## Rules

- **Resumable, always.** Checkpoint every N steps to `runs/<spec>/ckpt/`, and resume from the latest
  on start. Walltime kills and pre-emption are normal, not exceptional; losing eight hours to one is
  avoidable and infuriating.
- **Weights go to scratch, not home.** HF caches are tens of GB and home quotas are small.
- **Request honestly.** Over-requesting time/memory means queueing forever; under-requesting means a
  kill at hour seven. Estimate from the overfit run's step time × steps, then add ~30%.
- **Log to W&B** so the run can be watched without SSH.
- Write **stdout and stderr to `runs/`**, which is gitignored.

## Hand-off

Print the script and the exact `sbatch scripts/slurm/<spec>.sbatch`. Tell the user what to watch:
`squeue -u $USER`, the W&B curves, and **the first 50 loss values** — if loss is flat there, kill the
job immediately rather than paying for twelve hours of a broken pipeline.

Do **not** run it. Then append the `progress_report.md` entry recording what was queued and why.
