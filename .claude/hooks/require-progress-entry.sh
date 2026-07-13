#!/usr/bin/env bash
# Stop: a change is not done until progress_report.md records it (CLAUDE.md section 5).
#
# If anything under src/ specs/ configs/ tests/ scripts/ is newer than progress_report.md,
# the agent changed the project without telling the story. Block the stop and say so.
# Exit 2 = block + feed the message back to the agent.
set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0

# Don't loop: if we already blocked once this turn, let the agent stop.
already="$(python3 -c '
import json, sys
try:
    print(json.load(sys.stdin).get("stop_hook_active", False))
except Exception:
    print(False)
' 2>/dev/null)"
[[ "$already" == "True" ]] && exit 0

report="progress_report.md"
[[ -f "$report" ]] || exit 0

stale="$(find src specs configs tests scripts -type f \
          \( -name '*.py' -o -name '*.md' -o -name '*.yaml' -o -name '*.sh' \) \
          -newer "$report" 2>/dev/null | head -5)"

if [[ -n "$stale" ]]; then
  {
    echo "STOP BLOCKED: progress_report.md is stale."
    echo
    echo "These files changed after the last progress entry:"
    echo "$stale" | sed 's/^/  - /'
    echo
    echo "A change is not done until its story is recorded (CLAUDE.md section 5)."
    echo "Append a new [SEQ-000N] entry now — What / Why / How / Issues & resolutions / Verification."
    echo "Include what broke and how you fixed it; do not sanitise it."
  } >&2
  exit 2
fi

exit 0
