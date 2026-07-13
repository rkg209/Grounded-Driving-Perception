# Spec 00 — Scaffold

**Status:** done · **Compute:** laptop · **Depends on:** nothing

## Objective

A minimum working codebase: importable package, validated config, honest CLI, synthetic fixtures,
tests, and a green smoke script — so that every later spec has a place to land and a way to be
verified **offline, on the M4, with no dataset and no cluster**.

## Inputs / outputs

- **In:** nothing (no data, no weights, no network).
- **Out:** `uv run gdp info` prints resolved config + device; `make test`, `make smoke` pass.

## Approach

- `src/gdp/`: `config.py` (YAML → dataclass, deep-merged, **unknown keys rejected**), `paths.py`
  (repo-root-relative so laptop and cluster share one config), `seed.py` (determinism +
  `cuda → mps → cpu`), `logging.py`, `data.py` (COCO-style loader), `cli.py`.
- **CLI as roadmap:** the full command surface (`info`, `data`, `detect`, `evaluate`, `deploy`,
  `vqa`, `demo`) exists from day one. Only `info` runs; the rest exit 2 naming the spec that will
  implement them. The skeleton documents the project's true progress and cannot overstate it.
- **Synthetic mini-BDD fixture** (`scripts/make_fixtures.py` → `tests/fixtures/mini_bdd/`): four
  crude drawn road scenes whose ground truth is exact because we drew it. This is the keystone of
  the laptop-vs-cluster split (CLAUDE.md §4): it makes the data contract testable with zero
  downloads. Images are watermarked "NOT REAL DATA".

## Acceptance criteria

1. `uv sync` resolves; `uv run pytest` all green.
2. `uv run gdp info -c configs/default.yaml` emits valid JSON with device ∈ {cuda, mps, cpu}.
3. Every unimplemented CLI command exits 2 and names its spec.
4. `bash scripts/smoke.sh` exits 0 (import → config → device → fixture → CLI).
5. `uv run ruff check .` clean.
6. A typo'd config key raises, rather than being silently ignored.
7. `set_seed` makes python/numpy/torch draws reproducible.
8. No file in the repo claims a metric.

## Honesty contract

- **H7** — the fixture must be unmistakably synthetic (watermark + `info.description` in the
  annotations JSON + a test asserting it). **No metric computed on the fixture may ever be
  reported.** It tests plumbing, not perception.
- **H8** — the scaffold produces no numbers, so it can claim none. `runs/<spec>/<ts>/metrics.json`
  is established now as the *only* place results may originate.

## Out of scope

No model code, no Grounding-DINO, no Qwen, no dataset download, no ONNX, no training.

## Verification

`make install && make test && make lint && make smoke`
