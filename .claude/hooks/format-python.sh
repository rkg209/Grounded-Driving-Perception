#!/usr/bin/env bash
# PostToolUse(Edit|Write): keep Python formatted so lint never fails for cosmetic reasons.
set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0

file="$(python3 -c '
import json, sys
try:
    print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))
except Exception:
    print("")
' 2>/dev/null)"

[[ "$file" == *.py ]] || exit 0
[[ -f "$file" ]] || exit 0

command -v uv >/dev/null 2>&1 || exit 0
uv run ruff format "$file" >/dev/null 2>&1
uv run ruff check --fix "$file" >/dev/null 2>&1

exit 0
