#!/usr/bin/env bash
# Spec 10 task 10 — reproduce the laptop half of this project from a clean clone.
#
# Spec 10 §5: "an unreproduced repro section is a claim like any other". This script is how that
# claim gets discharged for everything the laptop can reach. It clones HEAD into a temporary
# directory — so it exercises what is *committed*, not what happens to be lying around in the
# working tree — and runs the whole offline path end to end, printing every step's exit code.
#
# It does NOT run any training or full-split evaluation: those need the cluster and are documented,
# with expected runtimes and labelled "not executed on this machine", in docs/reproduction.md.
#
# Usage:  bash scripts/repro_clean_clone.sh [--keep]
set -u

KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

SRC_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/gdp-repro.XXXXXX")"
CLONE="$WORK/clone"

cleanup() {
    if [ "$KEEP" = "1" ]; then
        echo "clone kept at: $CLONE"
    else
        rm -rf "$WORK"
    fi
}
trap cleanup EXIT

FAILED=0

step() {
    local name="$1"
    shift
    echo ""
    echo "=== $name"
    echo "--- \$ $*"
    ( cd "$CLONE" && "$@" )
    local rc=$?
    echo "--- exit code: $rc"
    [ "$rc" -ne 0 ] && FAILED=$((FAILED + 1))
    return 0
}

echo "=== clone HEAD of $SRC_REPO -> $CLONE"
git clone --quiet --no-hardlinks "$SRC_REPO" "$CLONE"
echo "--- exit code: $?"
echo "--- HEAD: $(git -C "$CLONE" rev-parse --short HEAD) $(git -C "$CLONE" log -1 --format=%s)"

step "install (uv sync)" make install
step "tests (pytest, model_heavy deselected)" make test
step "smoke (import -> config -> device -> fixture -> CLI)" make smoke
step "Stage-1 fixture data pipeline" \
    uv run gdp data prepare -c configs/default.yaml --dataset mini_bdd --split val
step "Stage-2 fixture data pipeline" \
    uv run gdp data prepare-drivelm -c configs/default.yaml --dataset mini_drivelm --split both
step "README tables match the committed snapshot" uv run gdp report render --check

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "=== ALL STEPS PASSED (laptop path only — the cluster path is documented, not run)"
else
    echo "=== $FAILED STEP(S) FAILED"
fi
exit "$FAILED"
