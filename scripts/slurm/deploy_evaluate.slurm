#!/bin/bash
#SBATCH --job-name=gdp-05-deploy-eval
#SBATCH --partition=gpu            # CPU-only work, but the `small` partition hangs at Python
                                    # startup for this repo (progress_report.md [SEQ-0141]/[SEQ-0142]);
                                    # `gpu` nodes are the ones verified to start reliably.
#SBATCH --gres=gpu:1               # Rudra's `gpu` partition QOS refuses to schedule a job with
                                    # zero GRES (job 415184 sat PD/QOSMinGRES until this was added,
                                    # progress_report.md [SEQ-0158]) — reserved but never touched:
                                    # `deploy bench`/`deploy evaluate-variants` pin the PyTorch
                                    # variant to device="cpu" unconditionally, so this job never
                                    # calls into CUDA.
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00            # bumped from 08:00:00 after TWO timeouts at exactly 8h
                                    # (jobs 415185, 416825 — progress_report.md [SEQ-0159]/
                                    # [SEQ-0160]): fp32-pt alone took ~4.4h, and fp32-onnx alone
                                    # did not finish within a further 8h on a dedicated A100 node's
                                    # CPU — this is real, measured cost, not a contention artifact.
                                    # `gpu` partition allows up to 4 days, so 24h has headroom.
#SBATCH --output=runs/05-deploy/slurm-%j.out
#SBATCH --error=runs/05-deploy/slurm-%j.err
#
# BEFORE SUBMITTING: `mkdir -p runs/05-deploy` (SLURM cannot create --output's directory and
# fails silently if it's missing). Requires runs/05-deploy/<ts>/{model.onnx,model.int8.onnx,
# export_result.json} already present (scp'd from the laptop's Action 4 Part A run,
# progress_report.md [SEQ-0156]) and `git pull`ed past commit 7f7cfa7 (the device-pinning fix AND
# the per-variant resume — without the former this job would silently run "fp32-pt" on this
# node's GPU instead of CPU; without the latter, every resubmission redoes every variant already
# on disk from scratch).
set -euo pipefail

# Action 4 Part B (REMAINING_ACTION_PLAN.md): compute real mAP for all three spec-05 variants
# (fp32-pt, fp32-onnx, int8-onnx) against the FULL real 10,000-image BDD100K val split, using the
# spec-03 fine-tuned checkpoint. The laptop's M4 CPU already produced the paired *latency* numbers
# (that half must stay on the M4 — H3, the edge-target claim) in the same run directory; mAP is a
# property of the model's outputs, not of the chip that computed it, so it does not need to run on
# the same hardware, and the laptop could not finish this step in reasonable time (contended CPU,
# see [SEQ-0156]).
#
# Resumable PER COMPLETED VARIANT ONLY (src/gdp/deploy/evaluate.py, [SEQ-0159]): a resubmission
# skips any variant whose predictions_<variant>.json already exists in the run dir, instead of
# redoing it. It is NOT resumable *within* a variant still in progress when the job dies — there is
# no mid-variant checkpointing (matching `gdp detect`'s own detect-then-write-once pattern; adding
# incremental checkpointing here would be new machinery this codebase has no precedent for). A job
# killed mid-`fp32-onnx` loses that variant's progress entirely and starts it over on resubmission.

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/cluster_env.sh

echo "==> git commit: $(cat .git/refs/heads/main 2>/dev/null || echo unknown)" >&2

CFG="-c configs/default.yaml -c configs/bdd100k.yaml -c configs/finetuned_checkpoint.yaml"

echo "==> gdp deploy evaluate-variants $CFG --dataset bdd100k --split val" >&2
uv run gdp deploy evaluate-variants $CFG --dataset bdd100k --split val

echo "==> DONE. See runs/05-deploy/<ts>/metrics.json and curve.png" >&2
