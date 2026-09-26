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
#SBATCH --time=08:00:00            # generous, unmeasured: real per-image cost on Rudra's CPU is
                                    # unknown (only the contended M4 laptop has been timed: ~2.8s
                                    # fp32-pt + ~3.0s fp32-onnx + ~3.0s int8-onnx per image)
#SBATCH --output=runs/05-deploy/slurm-%j.out
#SBATCH --error=runs/05-deploy/slurm-%j.err
#
# BEFORE SUBMITTING: `mkdir -p runs/05-deploy` (SLURM cannot create --output's directory and
# fails silently if it's missing). Requires runs/05-deploy/<ts>/{model.onnx,model.int8.onnx,
# export_result.json} already present (scp'd from the laptop's Action 4 Part A run,
# progress_report.md [SEQ-0156]) and `git pull`ed past commit f1f86d0 (the device-pinning fix —
# without it this job would silently run "fp32-pt" on this node's GPU instead of CPU).
set -euo pipefail

# Action 4 Part B (REMAINING_ACTION_PLAN.md): compute real mAP for all three spec-05 variants
# (fp32-pt, fp32-onnx, int8-onnx) against the FULL real 10,000-image BDD100K val split, using the
# spec-03 fine-tuned checkpoint. The laptop's M4 CPU already produced the paired *latency* numbers
# (that half must stay on the M4 — H3, the edge-target claim) in the same run directory; mAP is a
# property of the model's outputs, not of the chip that computed it, so it does not need to run on
# the same hardware, and the laptop could not finish this step in reasonable time (contended CPU,
# see [SEQ-0156]).
#
# NOT resumable: `gdp deploy evaluate-variants` re-detects and overwrites all three variants'
# predictions every time it runs, with no partial-completion check (src/gdp/deploy/evaluate.py has
# no cache/skip logic, unlike the run_step() pattern in zeroshot_eval.slurm/finetune_detector.slurm).
# A preempted or timed-out job must be resubmitted from scratch, not resumed. If this turns out to
# take multiple attempts, the fix is to add per-variant resume logic to evaluate.py before
# resubmitting a third time — do not just keep resubmitting blindly.

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/cluster_env.sh

echo "==> git commit: $(cat .git/refs/heads/main 2>/dev/null || echo unknown)" >&2

CFG="-c configs/default.yaml -c configs/bdd100k.yaml -c configs/finetuned_checkpoint.yaml"

echo "==> gdp deploy evaluate-variants $CFG --dataset bdd100k --split val" >&2
uv run gdp deploy evaluate-variants $CFG --dataset bdd100k --split val

echo "==> DONE. See runs/05-deploy/<ts>/metrics.json and curve.png" >&2
