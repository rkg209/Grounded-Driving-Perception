#!/bin/bash
#SBATCH --job-name=gdp-02-zeroshot
#SBATCH --partition=gpu            # EDIT: set to the IITB cluster's actual GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00            # inference-only (grounding-dino-tiny, no training) — generous
#SBATCH --output=runs/02-zeroshot/slurm-%j.out
#SBATCH --error=runs/02-zeroshot/slurm-%j.err
set -euo pipefail

# Spec 02 (zero-shot baseline, H8): the pretrained, unmodified Grounding-DINO's mAP on the full
# BDD100K val split. This is the ONE number every later Stage-1 claim (spec 03's fine-tuned mAP
# delta) is measured against, so it is produced here, on the full split, before any fine-tuning.
#
# Chain (per specs/02-zeroshot-baseline.md §5, "the whole spec" in one sentence):
#   data prepare (real bdd100k)
#     -> detect --split train (low threshold, headroom for the sweep)
#     -> evaluate --split train --sweep   (picks the operating threshold; chosen_on=train)
#     -> detect --split val --box-threshold <sweep winner>
#     -> evaluate --split val --box-threshold <winner> --chosen-on train   <-- THE H8 BASELINE
#
# Resumable: SLURM pre-emption/walltime kills are normal on a shared cluster, not exceptional.
# Each step's result path is cached in $STATE_DIR after it succeeds; a re-submitted job skips
# every already-completed step instead of re-running (and re-paying for) it.

cd "$SLURM_SUBMIT_DIR"
export HF_HOME="${SCRATCH:-$HOME}/hf"   # grounding-dino-tiny's weights are small, but never fill $HOME quota
export TOKENIZERS_PARALLELISM=false

CFG="-c configs/default.yaml -c configs/bdd100k.yaml"
STATE_DIR="runs/02-zeroshot/slurm-state"
mkdir -p "$STATE_DIR" data/bdd100k/labels/det_20

# How many train images the threshold sweep runs against — a representative slice, not the full
# ~70k-image train split, since the sweep only needs to pick one number.
SWEEP_LIMIT="${SWEEP_LIMIT:-2000}"
# The sweep only needs candidates >= 0.15 (configs/default.yaml's detector.sweep_candidates); this
# is set below that floor so the train predictions carry every score the sweep might pick.
SWEEP_HEADROOM_THRESHOLD="0.05"

# run_step NAME CMD...
#   Runs CMD (unless NAME's result is already cached), tees its output to a log for debugging,
#   and prints CMD's own "-> <path>" trailer (every `gdp detect`/`gdp evaluate` invocation ends
#   its summary line that way) to stdout — and ONLY that, so `X=$(run_step ...)` captures a
#   clean path. Everything else (progress, the command's full output) goes to stderr / the log,
#   which SLURM's --error file also captures.
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

# --- 1. Convert raw BDD100K labels into the project's COCO-style schema (spec 01), both splits.
# `data prepare` always writes to a *fresh* runs/01-data/<ts>/, but `detect`/`evaluate` read from
# configs/bdd100k.yaml's fixed `dataset.annotations` path (data/bdd100k/labels/det_20/
# det_val_coco.json, with 'val'->'train' swapped for the train split — see cli.py's
# `_detect_dataset_paths`). Those two conventions don't line up for real bdd100k the way they do
# for the mini_bdd fixture (whose annotations path already *is* the fixed fixture path), so this
# job copies the freshly converted files into the fixed location `detect`/`evaluate` expect.
DATA_MARKER="$STATE_DIR/data_prepare_dir.path"
if [[ ! -f "$DATA_MARKER" ]]; then
  echo "==> data_prepare: uv run gdp data prepare $CFG --dataset bdd100k --split both" >&2
  data_log="$STATE_DIR/data_prepare.log"
  if ! uv run gdp data prepare $CFG --dataset bdd100k --split both > "$data_log" 2>&1; then
    echo "FATAL: data_prepare failed — see $data_log" >&2
    tail -n 40 "$data_log" >&2
    exit 1
  fi
  cat "$data_log" >&2
  coco_path=$(grep -oE 'runs/01-data/[^ ]+det_train_coco\.json' "$data_log" | head -1)
  if [[ -z "$coco_path" ]]; then
    echo "FATAL: could not find det_train_coco.json in $data_log" >&2
    exit 1
  fi
  dirname "$coco_path" > "$DATA_MARKER"
fi
DATA_DIR=$(cat "$DATA_MARKER")
cp "$DATA_DIR/det_train_coco.json" data/bdd100k/labels/det_20/det_train_coco.json
cp "$DATA_DIR/det_val_coco.json" data/bdd100k/labels/det_20/det_val_coco.json

# --- 2. Detect on a train slice, with headroom below every sweep candidate.
PRED_TRAIN=$(run_step detect_train \
  uv run gdp detect $CFG --dataset bdd100k --split train \
    --limit "$SWEEP_LIMIT" --box-threshold "$SWEEP_HEADROOM_THRESHOLD")

# --- 3. Sweep the operating threshold on train ONLY (H8 leakage guard — never on val).
METRICS_TRAIN=$(run_step evaluate_train_sweep \
  uv run gdp evaluate $CFG --dataset bdd100k --split train \
    --predictions "$PRED_TRAIN" --sweep)

WINNER=$(uv run python -c "
import json, sys
print(json.load(open(sys.argv[1]))['box_threshold'])
" "$METRICS_TRAIN")
echo "==> sweep picked box_threshold=$WINNER (chosen_on=train, from $METRICS_TRAIN)" >&2

# --- 4. Detect on the FULL val split at the sweep-picked threshold.
PRED_VAL=$(run_step detect_val \
  uv run gdp detect $CFG --dataset bdd100k --split val --box-threshold "$WINNER")

# --- 5. THE H8 BASELINE: evaluate on val, replaying the train-picked threshold honestly
# (--chosen-on train records that this number was NOT tuned on val — it was chosen on train and
# is only being applied here).
METRICS_VAL=$(run_step evaluate_val \
  uv run gdp evaluate $CFG --dataset bdd100k --split val \
    --predictions "$PRED_VAL" --box-threshold "$WINNER" --chosen-on train)

echo "==> DONE. Zero-shot baseline (H8): $METRICS_VAL" >&2
cat "$METRICS_VAL"
