#!/bin/bash
#SBATCH --job-name=gdp-04-grounding
#SBATCH --partition=gpu            # EDIT: set to the IITB cluster's actual GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00            # EDIT: ~1 forward pass/phrase x 2 checkpoints; 150-300 phrases
                                    # is trivial GPU time, so this is dominated by queue + startup.
#SBATCH --output=runs/04-grounding/slurm-%j.out
#SBATCH --error=runs/04-grounding/slurm-%j.err
#
# BEFORE SUBMITTING, on Rudra: `mkdir -p runs/04-grounding` (SLURM cannot create the --output
# directory and fails the job silently if it is missing), and apply nothing else — the cluster
# fixes live in scripts/slurm/cluster_env.sh, sourced below.
set -euo pipefail

# Spec 04 (curated grounding evaluation set, H8/H9): score the SAME frozen, hash-locked phrase
# set against BOTH the zero-shot and fine-tuned detector, at the SAME replayed box_threshold, and
# refuse to emit a comparison unless they agree. Chain (per specs/04-grounding-eval-set.md, design
# decisions 6, 7, 9):
#
#   ground evaluate --checkpoint <zero-shot>   <-- REFUSES if phrases.json isn't frozen/unchanged
#     -> ground evaluate --checkpoint <fine-tuned>
#     -> ground compare                         <-- H8: refuses unless both agree on
#                                                    phrases_sha256/num_phrases/box_threshold
#
# Precondition, done by hand, NOT by this job (spec's own "Not in this plan" note): the 150-300
# real phrases have already been authored against `gdp ground sample-frames`'s overlays, then
#   gdp ground validate --phrases data/grounding_eval/phrases.json --dataset bdd100k
#   gdp ground freeze    --phrases data/grounding_eval/phrases.json
# have already run, producing data/grounding_eval/phrases.lock.json. This job REFUSES (via
# `assert_frozen`) if that lock is missing or the phrases file has been edited since.
#
# Resumable: SLURM pre-emption/walltime kills are normal on a shared cluster (same reasoning as
# scripts/slurm/finetune_detector.slurm). Each step's result path is cached in $STATE_DIR after it
# succeeds; a re-submitted job skips every already-completed step.

cd "$SLURM_SUBMIT_DIR"
# Cluster env (scratch paths, HF cache, offline flags, C headers, uv on PATH, provenance echo).
# See scripts/slurm/cluster_env.sh — it degrades to sane defaults off Rudra.
source scripts/slurm/cluster_env.sh

CFG="-c configs/default.yaml -c configs/bdd100k.yaml -c configs/grounding_eval.yaml"
STATE_DIR="runs/04-grounding/slurm-state"
mkdir -p "$STATE_DIR"

PHRASES="${PHRASES:-data/grounding_eval/phrases.json}"

# The H8 baseline this run's fine-tuned checkpoint sits next to, and the source of the ONE
# threshold both `ground evaluate` calls below replay (design decision 7 — never swept on this
# set). EDIT before submitting: there is no safe default, since pointing at the wrong run would
# silently produce an incomparable delta.
: "${ZEROSHOT_METRICS:?set ZEROSHOT_METRICS to spec 02's authoritative runs/02-zeroshot/<ts>/metrics.json (the val H8 baseline) before submitting}"
: "${FINETUNED_CHECKPOINT:?set FINETUNED_CHECKPOINT to spec 03's runs/03-finetune/<ts>/checkpoint-N before submitting}"

# run_step NAME CMD... — identical helper to finetune_detector.slurm/zeroshot_eval.slurm: skips
# CMD if NAME's result is already cached, tees output to a log, and prints only CMD's own
# "-> <path>" trailer to stdout.
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

# --- 0. Replay spec 02's train-chosen threshold — never re-sweep here (design decision 7).
WINNER=$(uv run python -c "
import json, sys
print(json.load(open(sys.argv[1]))['box_threshold'])
" "$ZEROSHOT_METRICS")
echo "==> replaying spec 02's threshold box_threshold=$WINNER (chosen_on=train) from $ZEROSHOT_METRICS" >&2

# --- 1. Zero-shot grounding accuracy. `ground evaluate` itself refuses (assert_frozen) unless
# $PHRASES still hashes to its committed <stem>.lock.json — the mechanical half of "no phrase was
# edited after seeing a prediction" (acceptance 5).
ZS_METRICS=$(run_step ground_evaluate_zeroshot \
  uv run gdp ground evaluate $CFG --dataset bdd100k --phrases "$PHRASES" \
    --checkpoint "IDEA-Research/grounding-dino-tiny" \
    --box-threshold "$WINNER" --chosen-on train)

# --- 2. Fine-tuned grounding accuracy — identical phrase set, identical threshold.
FT_METRICS=$(run_step ground_evaluate_finetuned \
  uv run gdp ground evaluate $CFG --dataset bdd100k --phrases "$PHRASES" \
    --checkpoint "$FINETUNED_CHECKPOINT" \
    --box-threshold "$WINNER" --chosen-on train)

# --- 3. THE H8 DELTA: refuses to emit unless both metrics.json's agree on
# phrases_sha256/num_phrases/box_threshold (design decision 9).
COMPARISON=$(run_step ground_compare \
  uv run gdp ground compare --zeroshot "$ZS_METRICS" --finetuned "$FT_METRICS")

echo "==> DONE. Grounding accuracy delta (H8, self-built set — H9): $COMPARISON" >&2
cat "$COMPARISON"
