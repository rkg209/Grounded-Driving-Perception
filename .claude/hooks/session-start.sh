#!/usr/bin/env bash
# SessionStart: orient the agent in the SDD loop before it does anything.
set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0

echo "=== Grounded Driving Perception — session start ==="
echo
echo "Read CLAUDE.md before acting. Honesty Register H1-H8 is the highest law of this repo."
echo "Training NEVER runs in-session (M4 laptop). Emit a SLURM script via /train-job."
echo

if [[ -f specs/README.md ]]; then
  echo "--- Spec status ---"
  grep -E '^\| [0-9]{2} \|' specs/README.md | awk -F'|' '{gsub(/^ +| +$/,"",$2); gsub(/^ +| +$/,"",$3); gsub(/^ +| +$/,"",$4); printf "  %s %-28s %s\n", $2, $3, $4}'
  echo
fi

if [[ -f progress_report.md ]]; then
  last=$(grep -E '^## \[SEQ-' progress_report.md | tail -1)
  echo "--- Last progress entry ---"
  echo "  ${last:-none yet}"
  echo
  echo "REMINDER: every change must append a new entry to progress_report.md (What/Why/How/Issues/Verification)."
fi

exit 0
