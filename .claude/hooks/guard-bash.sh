#!/usr/bin/env bash
# PreToolUse(Bash): block the three things that would quietly wreck this project.
#   1. destructive deletes
#   2. committing datasets / weights (they are huge and not redistributable)
#   3. launching a full training run in-session (the M4 cannot do it; use /train-job)
# Exit 2 = block the call and show the message to the agent.
set -uo pipefail

cmd="$(python3 -c '
import json, sys
try:
    print(json.load(sys.stdin).get("tool_input", {}).get("command", ""))
except Exception:
    print("")
' 2>/dev/null)"

[[ -z "$cmd" ]] && exit 0

deny() {
  echo "BLOCKED by guard-bash hook: $1" >&2
  echo "$2" >&2
  exit 2
}

# 1. Destructive deletes
if [[ "$cmd" =~ rm[[:space:]]+(-[a-zA-Z]*[rf][a-zA-Z]*[[:space:]]+)+(/|~|\.|\*) ]]; then
  deny "recursive/forced delete of a broad path." \
       "Delete specific files explicitly, or ask the user to do it."
fi

# 2. Committing data, weights, or run outputs
if [[ "$cmd" =~ git[[:space:]]+add ]] && [[ "$cmd" =~ (data/|runs/|\.onnx|\.pt|\.pth|\.safetensors|checkpoints/) ]]; then
  deny "attempt to git-add datasets, weights, or run outputs." \
       "These are gitignored on purpose: BDD100K/nuScenes are not redistributable and weights are huge. Commit code and specs only."
fi

# 3. Full training in-session
if [[ "$cmd" =~ (gdp[[:space:]]+train|train\.py|accelerate[[:space:]]+launch|torchrun|deepspeed|sbatch) ]]; then
  deny "attempt to launch training (or sbatch) from the session." \
       "The M4 laptop cannot fine-tune Grounding-DINO or a 3B VLM (CLAUDE.md section 4).
Use /train-job to emit a SLURM script and hand it to the user, who launches it themselves.
Exception: a tiny overfit sanity check is fine, but run it via pytest, not a train entrypoint."
fi

exit 0
