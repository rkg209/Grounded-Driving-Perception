#!/bin/bash
#SBATCH --job-name=gdp-03-finetune
#SBATCH --partition=gpu            # EDIT: set to the IITB cluster's actual GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00            # EDIT: estimate from the overfit gate's step time x total
                                    # steps (training.epochs x steps-per-epoch in
                                    # configs/train_detector.yaml) + ~30%, then requeue if it
                                    # still doesn't finish — see the resume note below.
#SBATCH --output=runs/03-finetune/slurm-%j.out
#SBATCH --error=runs/03-finetune/slurm-%j.err
set -euo pipefail

# Spec 03 (fine-tune the detector, H1/H8): fine-tune Grounding-DINO on BDD100K train and report
# the fine-tuned mAP NEXT TO spec 02's zero-shot mAP on the identical val split, at the identical
# operating threshold. Chain (per specs/03-finetune-detector.md, design decisions 5 and 7):
#
#   overfit gate (--overfit N)      <-- REFUSES to let the full run start if this fails
#     -> train detector (full run)
#     -> detect --split val --checkpoint <fine-tuned> --box-threshold <spec 02's winner>
#     -> evaluate --split val --checkpoint ... --chosen-on train        <-- the H8 pairing
#     -> compare --zeroshot <spec 02 metrics.json> --finetuned <the above>
#     -> probe-openvocab, once per checkpoint (H9: demo only, never a metric)
#
# Resumable: SLURM pre-emption/walltime kills are normal on a shared cluster (same reasoning as
# scripts/slurm/zeroshot_eval.slurm). Each step's result path is cached in $STATE_DIR after it
# succeeds; a re-submitted job skips every already-completed step. The one caveat: `gdp train
# detector` always writes to a FRESH runs/03-finetune/<ts>/ (the CLI has no --out-dir flag), so a
# job killed mid-training and resubmitted resumes the OPTIMIZER STEP from the latest checkpoint-N
# it can find on disk (via --resume), but that continuation lands in a new timestamped directory
# rather than the original one. This is a known, documented gap, not a silent one: watch
# $STATE_DIR/train_full.log across resubmissions if you need every checkpoint in one place.

cd "$SLURM_SUBMIT_DIR"
export WANDB_PROJECT=grounded-driving-perception   # WANDB_API_KEY, if set, turns on curve logging
export HF_HOME="${SCRATCH:-$HOME}/hf"               # never fill $HOME quota with weights
export TOKENIZERS_PARALLELISM=false

CFG="-c configs/default.yaml -c configs/bdd100k.yaml -c configs/train_detector.yaml"
STATE_DIR="runs/03-finetune/slurm-state"
mkdir -p "$STATE_DIR" data/bdd100k/labels/det_20

# The H8 baseline this run's fine-tuned mAP will sit next to: spec 02's val metrics.json, chosen
# on train, never re-swept. EDIT this to the actual path before submitting — there is no safe
# default because pointing at the wrong run silently produces an incomparable delta (H8).
: "${ZEROSHOT_METRICS:?set ZEROSHOT_METRICS to spec 02 authoritative runs/02-zeroshot/<ts>/metrics.json (the val H8 baseline) before submitting}"

# run_step NAME CMD... — identical helper to zeroshot_eval.slurm: skips CMD if NAME's result is
# already cached, tees output to a log, and prints only CMD's own "-> <path>" trailer to stdout.
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

# --- 0. BDD100K train/val must already be in the fixed location detect/evaluate read (same
# copy-into-place dance as zeroshot_eval.slurm's step 1 — skipped here if already present, since
# spec 02's job normally runs first and leaves these files behind).
if [[ ! -f data/bdd100k/labels/det_20/det_train_coco.json ]]; then
  echo "==> data_prepare: uv run gdp data prepare $CFG --dataset bdd100k --split both" >&2
  data_log="$STATE_DIR/data_prepare.log"
  if ! uv run gdp data prepare $CFG --dataset bdd100k --split both > "$data_log" 2>&1; then
    echo "FATAL: data_prepare failed — see $data_log" >&2
    tail -n 40 "$data_log" >&2
    exit 1
  fi
  cat "$data_log" >&2
  coco_path=$(grep -oE 'runs/01-data/[^ ]+det_train_coco\.json' "$data_log" | head -1)
  [[ -z "$coco_path" ]] && { echo "FATAL: could not find det_train_coco.json in $data_log" >&2; exit 1; }
  data_dir=$(dirname "$coco_path")
  cp "$data_dir/det_train_coco.json" data/bdd100k/labels/det_20/det_train_coco.json
  cp "$data_dir/det_val_coco.json" data/bdd100k/labels/det_20/det_val_coco.json
fi

# --- 1. Overfit-20 gate (design decision 5, acceptance 1): REFUSES to let the full run start
# below. `gdp train detector --overfit` exits non-zero with "overfit gate FAILED" if the final
# loss doesn't clear training.overfit_loss_target — run_step treats that as fatal, same as any
# other failed step, which is exactly the "the SLURM job refuses to start the full run" contract.
run_step overfit_gate \
  uv run gdp train detector $CFG --dataset bdd100k --split train \
    --overfit "${OVERFIT_IMAGES:-20}" --max-steps "${OVERFIT_MAX_STEPS:-200}"

# --- 2. The real fine-tuning run. If a checkpoint from a previously-killed attempt exists on
# disk, resume from its latest step rather than restarting at 0 (see the resumability note above).
LATEST_CKPT=$(ls -dt runs/03-finetune/*/checkpoint-* 2>/dev/null | head -1 || true)
RESUME_ARGS=()
if [[ -n "$LATEST_CKPT" ]]; then
  echo "==> resuming fine-tuning from $LATEST_CKPT" >&2
  RESUME_ARGS=(--resume "$LATEST_CKPT")
fi
CKPT_FINAL=$(run_step train_full \
  uv run gdp train detector $CFG --dataset bdd100k --split train "${RESUME_ARGS[@]}")

# --- 3. Replay spec 02's train-chosen threshold — never re-sweep on the fine-tuned model (that
# would make the two mAPs incomparable and re-open the leakage question, design decision 7).
WINNER=$(uv run python -c "
import json, sys
print(json.load(open(sys.argv[1]))['box_threshold'])
" "$ZEROSHOT_METRICS")
echo "==> replaying spec 02's threshold box_threshold=$WINNER (chosen_on=train) from $ZEROSHOT_METRICS" >&2

# --- 4. Re-score the fine-tuned checkpoint with the IDENTICAL detect/evaluate code as spec 02
# (design decision 7 — this identity is what makes the mAP delta honest).
PRED_VAL=$(run_step detect_val_finetuned \
  uv run gdp detect $CFG --dataset bdd100k --split val \
    --checkpoint "$CKPT_FINAL" --run-spec 03-finetune --box-threshold "$WINNER")

METRICS_VAL=$(run_step evaluate_val_finetuned \
  uv run gdp evaluate $CFG --dataset bdd100k --split val \
    --predictions "$PRED_VAL" --run-spec 03-finetune \
    --box-threshold "$WINNER" --chosen-on train)

# --- 5. THE H8 DELTA: refuses to emit unless both metrics.json's agree on
# dataset/split/num_images/box_threshold (design decision 8).
COMPARISON=$(run_step compare \
  uv run gdp compare --zeroshot "$ZEROSHOT_METRICS" --finetuned "$METRICS_VAL")

echo "==> DONE. Fine-tuned mAP delta (H8): $COMPARISON" >&2
cat "$COMPARISON"

# --- 6. Open-vocab forgetting probe (design decision 9, H9): a demo, labelled as one — no
# accuracy number, ever. Run once per checkpoint so the write-up can compare qualitatively.
PROBE_IMAGES="${PROBE_IMAGES:-data/bdd100k/images/100k/val}"
run_step probe_zeroshot \
  uv run gdp probe-openvocab $CFG --checkpoint "IDEA-Research/grounding-dino-tiny" \
    --images-root "$PROBE_IMAGES" --limit "${PROBE_LIMIT:-8}"
run_step probe_finetuned \
  uv run gdp probe-openvocab $CFG --checkpoint "$CKPT_FINAL" \
    --images-root "$PROBE_IMAGES" --limit "${PROBE_LIMIT:-8}"
