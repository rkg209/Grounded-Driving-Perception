#!/bin/bash
#SBATCH --job-name=gdp-08-vqa-eval
#SBATCH --partition=gpu            # EDIT: set to the IITB cluster's actual GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00            # EDIT: estimate from val split size x per-item generation time
                                    # (two full predict passes: base + finetuned) + ~30%, then
                                    # requeue if it still doesn't finish — see the resume note.
#SBATCH --output=runs/08-vqa/slurm-%j.out
#SBATCH --error=runs/08-vqa/slurm-%j.err
set -euo pipefail

# Spec 08 (base-vs-fine-tuned VQA accuracy on DriveLM's official split, H2): scores spec 07's
# adapter against the frozen base model with the vendored official scorer. Chain:
#
#   predict_base        <-- GPU, expensive, resumable at item granularity
#     -> predict_finetuned
#     -> score           <-- CPU-cheap; the only step that calls the vendored scorer
#     -> failures         <-- samples + scaffolds failures.md for human annotation
#
# No training happens here — spec 07 owns that (scripts/slurm/finetune_vlm.slurm). This script
# consumes its adapter output verbatim.
#
# Resumable: identical `run_step` idiom to scripts/slurm/finetune_vlm.slurm / finetune_detector.slurm.
# `gdp vqa predict`'s own item-level resume (spec 08 design decision 5) means a wall-clock kill
# mid-`predict_base`/`predict_finetuned` loses at most the in-flight item when the job is
# resubmitted with the same PREDICTIONS_BASE/PREDICTIONS_FINETUNED paths — set them explicitly
# (rather than letting them default) if you need that continuity across a requeue.

cd "$SLURM_SUBMIT_DIR"
export HF_HOME="${SCRATCH:-$HOME}/hf"               # never fill $HOME quota with the 3B model's weights
export TOKENIZERS_PARALLELISM=false

CFG="-c configs/default.yaml -c configs/drivelm.yaml -c configs/vqa_eval.yaml"
STATE_DIR="runs/08-vqa/slurm-state"
mkdir -p "$STATE_DIR"

# No safe defaults for either — pointing at the wrong val split or a stale adapter would silently
# produce a headline number that isn't the one this run claims to be.
: "${VAL_JSONL:?set VAL_JSONL to spec 06 runs/06-data/<ts>/val.jsonl (or data/drivelm/val.jsonl) before submitting}"
: "${ADAPTER:?set ADAPTER to spec 07 runs/07-finetune-vlm/<ts>/checkpoint-<step> before submitting}"

PREDICTIONS_BASE="${PREDICTIONS_BASE:-runs/08-vqa/predictions_base.jsonl}"
PREDICTIONS_FINETUNED="${PREDICTIONS_FINETUNED:-runs/08-vqa/predictions_finetuned.jsonl}"
mkdir -p "$(dirname "$PREDICTIONS_BASE")" "$(dirname "$PREDICTIONS_FINETUNED")"

# run_step NAME CMD... — identical helper to finetune_vlm.slurm/finetune_detector.slurm: skips CMD
# if NAME's result is already cached, tees output to a log, and prints only CMD's own
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

# --- 1. Base model predictions (no adapter — the frozen Qwen2.5-VL-3B-Instruct).
run_step predict_base \
  uv run gdp vqa predict $CFG --val-jsonl "$VAL_JSONL" \
    --model-role base --out "$PREDICTIONS_BASE"

# --- 2. Fine-tuned model predictions (spec 07's LoRA adapter on the same split, same decoding —
# H8 requires both runs share config, and `gdp vqa predict` pins decoding from vqa_eval.yaml for
# both regardless of --model-role, so this is enforced by construction, not by convention).
run_step predict_finetuned \
  uv run gdp vqa predict $CFG --val-jsonl "$VAL_JSONL" \
    --model-role finetuned --adapter "$ADAPTER" --out "$PREDICTIONS_FINETUNED"

# --- 3. Score both against the vendored official scorer (spec 08 tasks 4-7). Writes metrics.json
# ({"models": {"base", "finetuned"}}, each with its own hallucination block), per_item_base.jsonl,
# per_item_finetuned.jsonl, and comparison.json — refuses to emit unless both runs' split hash,
# item count, decoding settings, and scorer hash all agree (H8).
METRICS_JSON=$(run_step score \
  uv run gdp vqa score $CFG --val-jsonl "$VAL_JSONL" \
    --base-predictions "$PREDICTIONS_BASE" --finetuned-predictions "$PREDICTIONS_FINETUNED")

echo "==> DONE. metrics -> $METRICS_JSON" >&2

# --- 4. Sample + scaffold the failure analysis (spec 08 task 8, acceptance criterion 4). The
# `Category:`/`Comment:` fields are left empty for a human — never auto-generated (H7).
METRICS_DIR=$(dirname "$METRICS_JSON")
run_step failures \
  uv run gdp vqa failures $CFG --val-jsonl "$VAL_JSONL" \
    --per-item "$METRICS_DIR/per_item_finetuned.jsonl" \
    --base-per-item "$METRICS_DIR/per_item_base.jsonl"
