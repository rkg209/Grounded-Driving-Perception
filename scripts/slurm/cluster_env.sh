#!/bin/bash
# Cluster environment shared by every job script in this directory. Sourced, never executed.
#
# Why this file exists: the five job scripts were written on the laptop against a generic SLURM
# cluster, and PARAM Rudra breaks four of those assumptions in ways that kill a job minutes in
# (progress_report.md [SEQ-0122], PARAM_Rudra_GDP_Cluster_Guide.md §0.5). Putting the fixes here
# keeps them in one place and keeps the job scripts readable as *the spec's chain of steps*
# rather than as a pile of cluster workarounds.
#
# Every setting below degrades to a sane default on a cluster that isn't Rudra: nothing here is
# required for the scripts to run somewhere else, and $SCR/$SCRATCH/$HOME are tried in that order.

# --- Scratch. Rudra's per-user scratch is NOT /scratch/$USER, and $SCRATCH is not defined there;
# the account's own convention is $SCR, exported from ~/.bashrc — which a SLURM job does NOT
# source, so it has to be re-derived here.
if [[ -z "${SCR:-}" ]]; then
  for candidate in "/scratch/IITB/ai-at-ieor/$USER" "${SCRATCH:-}" "/scratch/$USER"; do
    if [[ -n "$candidate" && -d "$candidate" ]]; then
      SCR="$candidate"
      break
    fi
  done
fi
export SCR="${SCR:-$HOME}"

# --- Model weights. The original `${SCRATCH:-$HOME}/hf` resolved to $HOME on Rudra, whose group
# quota is FULL (1T/1T) — the job dies partway through downloading Grounding-DINO (~700 MB) or
# Qwen2.5-VL-3B (~7 GB) with "Disk quota exceeded". $HF_HOME is reused if already set so that a
# login-node pre-download and the job share one cache.
# An inherited value is kept ONLY if it is not under $HOME: a stale ~/.bashrc export pointing at
# home is exactly the failure this is guarding against, so deferring to it would defeat the point.
if [[ -z "${HF_HOME:-}" || "${HF_HOME}" == "$HOME"/* ]]; then
  export HF_HOME="$SCR/.cache/huggingface"
fi

# --- Compute nodes have NO internet (DNS fails outright). Every model must be pre-downloaded on
# a login node; these two vars turn a silent 30-minute hang into an immediate, honest error.
# Set GDP_ALLOW_NETWORK=1 to opt out on a cluster whose compute nodes are online.
if [[ "${GDP_ALLOW_NETWORK:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# --- Triton JIT-compiles on the first CUDA forward pass and needs C headers, which Rudra's
# compute nodes lack (/usr/include/stdlib.h absent). The sysroot copy is made once on a login
# node: cp -r /usr/include $SCR/sysroot/
if [[ -d "$SCR/sysroot/include" ]]; then
  export C_INCLUDE_PATH="$SCR/sysroot/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
fi

# --- uv lives on scratch and is only on PATH via ~/.bashrc, which jobs don't source. UV_NO_SYNC
# is load-bearing: `uv run` would otherwise re-sync the venv and silently undo the cu126 torch
# swap Rudra's CUDA 12.4 driver requires (a cu13x wheel cannot run there at all).
export PATH="$SCR/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCR/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SCR/.local/share/uv/python}"
export UV_NO_SYNC=1

# --- Caches and temp: never $HOME (full quota), and never /tmp (small, node-local, wiped).
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCR/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCR/.cache/triton}"
# SLURM often hands a job TMPDIR=/tmp (node-local, small, wiped at job end) — a 7 GB model
# download or an ONNX export lands there and fails on a full disk, far from where it started.
if [[ -z "${TMPDIR:-}" || "${TMPDIR}" == /tmp* || "${TMPDIR}" == "$HOME"/* ]]; then
  export TMPDIR="$SCR/tmp"
fi
mkdir -p "$TMPDIR" "$HF_HOME"

export TOKENIZERS_PARALLELISM=false

# --- Provenance. Compute nodes have no `git`, so `gdp.paths.git_sha()` used to fall back to
# "unknown" and every cluster artifact lost its commit (H8: a number needs its split, its
# baseline AND its code version). The Python side now reads .git directly; this echo puts the
# same facts in the job log, where they survive even if an artifact is never written.
echo "== job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "== commit $(cat .git/refs/heads/main 2>/dev/null || cat .git/HEAD 2>/dev/null || echo unknown)"
echo "== SCR=$SCR HF_HOME=$HF_HOME offline=${HF_HUB_OFFLINE:-0}"
if command -v nvidia-smi > /dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi
