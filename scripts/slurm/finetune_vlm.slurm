#!/bin/bash
#SBATCH --job-name=gdp-07-finetune-vlm
#SBATCH --partition=gpu            # EDIT: set to the IITB cluster's actual GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00            # EDIT: estimate from the overfit gate's step time x total
                                    # steps (vlm_training.epochs x steps-per-epoch in
                                    # configs/train_vlm.yaml) + ~30%, then requeue if it still
                                    # doesn't finish — see the resume note below.
#SBATCH --output=runs/07-finetune-vlm/slurm-%j.out
#SBATCH --error=runs/07-finetune-vlm/slurm-%j.err
#
# BEFORE SUBMITTING, on Rudra: `mkdir -p runs/07-finetune-vlm` (SLURM cannot create the --output
# directory and fails the job silently if it is missing), and apply nothing else — the cluster
# fixes live in scripts/slurm/cluster_env.sh, sourced below.
set -euo pipefail

# Spec 07 (LoRA fine-tune Qwen2.5-VL-3B on DriveLM, H1): trains the Stage-2 adapter. Chain:
#
#   overfit gate (--overfit N)      <-- REFUSES to let the full run start if this fails
#     -> train vlm (full run)
#     -> vqa generate (one fixture record, printed as a demo — never scored)
#
# Deliberately stops there: **no evaluation step** (H2). Base-vs-fine-tuned accuracy, per-category
# breakdown, and failure analysis are spec 08's job, on a separate invocation against this run's
# adapter — mixing that in here would blur "did it train?" into "is it better?", exactly what the
# spec's two-stage split (train here, eval in 08) exists to keep apart.
#
# Resumable: SLURM pre-emption/walltime kills are normal on a shared cluster (same reasoning as
# scripts/slurm/finetune_detector.slurm). Each step's result path is cached in $STATE_DIR after it
# succeeds; a re-submitted job skips every already-completed step. The one caveat, identical to
# spec 03's job: `gdp train vlm` always writes to a FRESH runs/07-finetune-vlm/<ts>/ (no --out-dir
# flag), so a job killed mid-training and resubmitted resumes the OPTIMIZER STEP from the latest
# checkpoint-N it can find on disk (via --resume), but that continuation lands in a new
# timestamped directory rather than the original one.

cd "$SLURM_SUBMIT_DIR"
export WANDB_PROJECT=gdp-finetune-vlm               # WANDB_API_KEY, if set, turns on curve logging
# Cluster env (scratch paths, HF cache, offline flags, C headers, uv on PATH, provenance echo).
# See scripts/slurm/cluster_env.sh — it degrades to sane defaults off Rudra.
source scripts/slurm/cluster_env.sh

CFG="-c configs/default.yaml -c configs/drivelm.yaml -c configs/train_vlm.yaml"
STATE_DIR="runs/07-finetune-vlm/slurm-state"
mkdir -p "$STATE_DIR"

# Spec 06's conversion output: DriveLM records opened relative to drivelm.nuscenes_root. No safe
# default — pointing at the wrong file would silently train on the wrong (or synthetic) split.
: "${TRAIN_JSONL:?set TRAIN_JSONL to spec 06 runs/06-data/<ts>/train.jsonl (or data/drivelm/train.jsonl) before submitting}"

# run_step NAME CMD... — identical helper to finetune_detector.slurm: skips CMD if NAME's result
# is already cached, tees output to a log, and prints only CMD's own "-> <path>" trailer to stdout.
run_step() {
  local name="$1"; shift
  local marker="$STATE_DIR/$name.path"
  if [[ -f "$marker" ]]; then
    echo "==> $name: already done" >&2
    cat "$marker"
    return 0
  fi
  local log="$STATE_DIR/$name.log"
  echo "==> $name: $*" >&2
  if ! "$@" > "$log" 2>&1; then
    echo "FATAL: $name failed — see $log" >&2
    tail -n 40 "$log" >&2
    exit 1
  fi
  cat "$log" >&2
  local out_path
  out_path=$(grep -oE -- '-> \S+' "$log" | tail -1 | sed 's/^-> //')
  if [[ -z "$out_path" ]]; then
    echo "FATAL: $name produced no '-> <path>' trailer in $log" >&2
    exit 1
  fi
  echo "$out_path" > "$marker"
  echo "$out_path"
}

# --- 1. Overfit-16 gate (design decision 5, acceptance 1): REFUSES to let the full run start
# below. The training command exits non-zero with "overfit gate FAILED" if the final loss doesn't
# clear vlm_training.overfit_loss_target — run_step treats that as fatal, same as any other failed
# step, which is exactly the "refuses to start the full run" contract.
run_step overfit_gate \
  uv run gdp train vlm $CFG --train-jsonl "$TRAIN_JSONL" \
    --overfit "${OVERFIT_QA_PAIRS:-16}" --max-steps "${OVERFIT_MAX_STEPS:-200}"

# --- 2. The real fine-tuning run. If a checkpoint from a previously-killed attempt exists on
# disk, resume from its latest step rather than restarting at 0 (see the resumability note above).
LATEST_CKPT=$(ls -dt runs/07-finetune-vlm/*/checkpoint-* 2>/dev/null | head -1 || true)
RESUME_ARGS=()
if [[ -n "$LATEST_CKPT" ]]; then
  echo "==> resuming fine-tuning from $LATEST_CKPT" >&2
  RESUME_ARGS=(--resume "$LATEST_CKPT")
fi
CKPT_FINAL=$(run_step train_full \
  uv run gdp train vlm $CFG --train-jsonl "$TRAIN_JSONL" "${RESUME_ARGS[@]}")

echo "==> DONE. Adapter -> $CKPT_FINAL" >&2

# --- 3. Generation check (acceptance criterion 4): a demo, labelled as one — no accuracy number,
# ever (H2). Proves the adapter loads back and produces coherent text on one fixture record; spec
# 08 owns every number that could be called a VQA result.
run_step generation_check \
  uv run gdp vqa generate $CFG --adapter "$CKPT_FINAL" --train-jsonl "$TRAIN_JSONL" --index 0
