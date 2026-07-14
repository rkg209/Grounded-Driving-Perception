# Plan — Spec 00: Scaffold

**Spec:** [`specs/00-scaffold.md`](../../specs/00-scaffold.md) · **Status:** done ·
**Compute:** laptop (entirely) · **Depends on:** nothing

> **Note on this document.** Spec 00 was implemented before the `.claude/plans/` convention existed,
> so this plan is written *retroactively, against the scaffold as actually built* — verified by
> re-running its acceptance checks, not from memory. It exists because 00 is the keystone spec:
> it establishes the config system, the path contract, the fixture, and the CLI-as-roadmap that
> every later spec inherits, and that should not be the one spec with no design document. It records
> no metric and changes no code.

---

## 1 · Objective and compute target

A minimum working codebase, so that every later spec has **a place to land and a way to be
verified** — offline, on the M4, with no dataset and no cluster.

**Compute target: laptop, for every step.** Nothing in spec 00 touches a GPU, a model weight, or
the network. That is what makes it the *anchor* of the laptop-vs-cluster rule (CLAUDE.md §4) rather
than a victim of it: the constraint that the repo must stay verifiable without the cluster is
discharged here, once, for everything that follows.

## 2 · Files as built, and what each is responsible for

The design decision each file encodes is the part a later spec actually needs to know.

| File | Responsibility | The decision it encodes |
|---|---|---|
| `src/gdp/paths.py` | `repo_root()`, `resolve()`, `run_dir(spec)` | Every path is repo-root-relative (found by walking up to `pyproject.toml`), so one config file works unchanged on the M4 and on SLURM. `run_dir()` mints `runs/<spec>/<timestamp>/` — the **only** place a result may originate (CLAUDE.md §8). |
| `src/gdp/config.py` | `Config` dataclass tree, `load_config(*paths)` | YAML → dataclass, deep-merged left-to-right (later wins), **unknown keys raise**. `BDD100K_CLASSES` lives here and its order is load-bearing: it fixes the class index used for Grounding-DINO's contrastive text targets. |
| `src/gdp/seed.py` | `set_seed()`, `select_device()` | Seeds python/numpy/torch. `auto` resolves `cuda → mps → cpu`; an **explicit** request for an unavailable device raises rather than falling back. |
| `src/gdp/data.py` | `Box`, `Sample`, `Dataset`, `load_dataset()` | The COCO-style data contract. Converts COCO `xywh` → internal `xyxy` at the loader boundary, and validates hard: degenerate box, unknown `category_id`, and missing image file all raise. |
| `src/gdp/logging.py` | `get_logger()` | One log format for laptop and cluster, so the two read the same. |
| `src/gdp/cli.py` | the `gdp` command surface | **CLI-as-roadmap** — see §3. |
| `scripts/make_fixtures.py` | draws the synthetic mini-BDD | Deterministic: same bytes every run (`make fixtures`). |
| `scripts/smoke.sh` | 5-step end-to-end liveness check | import → config → device → fixture → CLI. Offline; no weights, no network. |
| `configs/default.yaml` | laptop defaults, pointing at the fixture | The repo runs with zero downloads. |
| `configs/bdd100k.yaml` | overlay for the real dataset | Nothing in it works until spec 01 — **its paths *are* the contract spec 01 must satisfy**. |
| `tests/` | 34 tests: config, seed, cli, fixtures | See §5. |

## 3 · The two ideas specs 01+ depend on

### CLI-as-roadmap

The full command surface (`info`, `data`, `detect`, `evaluate`, `deploy`, `vqa`, `demo`) exists from
day one. **Only `info` runs.** Every other command calls `_pending(spec, what)`, which names the spec
that will implement it and **exits 2**.

The skeleton therefore documents the project's true progress and *cannot overstate it*: `gdp --help`
is an honest status report, and a command starts working exactly when its spec lands.
`tests/test_cli.py::test_unimplemented_commands_fail_loudly_and_name_their_spec` pins this per
command. **When spec 01 implements `gdp data`, removing that command's row from the parametrized
test is part of the work** — the test is the ratchet.

### The synthetic fixture is a data-contract test, not a perception test

Four crude 640×360 road scenes, 14 boxes, 10 categories, drawn by `make_fixtures.py` — so the ground
truth is exact *because we drew it*.

This is the keystone of the laptop-vs-cluster split: it makes the data contract testable with zero
downloads, which is what lets CLAUDE.md §4 demand that every later spec stay verifiable on
`tests/fixtures/`. It proves the pipeline plumbs images and boxes correctly. It proves **nothing**
about model accuracy, and **no metric computed on it may ever be reported** (H7).

## 4 · Acceptance criteria → what discharges each

| # | Criterion | Discharged by | Re-run result |
|---|---|---|---|
| 1 | `uv sync` resolves; `pytest` green | `tests/` | **34 passed** |
| 2 | `gdp info` emits valid JSON, device ∈ {cuda, mps, cpu} | `cli.info` | `"device": "mps"` |
| 3 | Every unimplemented command exits 2 and names its spec | `cli._pending` + `test_cli.py` | pass |
| 4 | `bash scripts/smoke.sh` exits 0 | `scripts/smoke.sh` | `SMOKE OK`, exit 0 |
| 5 | `ruff check .` clean | — | `All checks passed!` |
| 6 | A typo'd config key raises | `config._build` unknown-key check | `test_unknown_key_is_rejected` |
| 7 | `set_seed` makes draws reproducible | `seed.set_seed` | `test_set_seed_makes_runs_reproducible` |
| 8 | No file claims a metric | the scaffold computes none; `run_dir()` is the only origin of results | pass |

## 5 · Subtle failure modes the scaffold catches

Each of these is a real trap, and each has a test that fails loudly.

- **A typo'd config key, silently ignored.** `speed: 42` instead of `seed: 42` would run the entire
  project on the wrong seed and never say a word. `_build()` diffs the YAML keys against the
  dataclass fields and raises, naming the key and its path. *A silently-ignored typo is how a
  "result" becomes a lie.* → `test_unknown_key_is_rejected`
- **`from __future__ import annotations` breaking dataclass recursion.** With postponed annotations,
  `Field.type` is the *string* `"PathsConfig"`, so `is_dataclass()` on it is always false and nested
  config sections would silently be built as raw dicts. `_build()` reads `get_type_hints(cls)`
  instead. Easy to reintroduce; the comment in `config.py` says why.
  → `test_merge_is_deep_and_right_wins`
- **Silent CPU fallback.** Requesting `cuda` on a node without it and getting CPU means discovering
  after ten hours that the run was worthless. `select_device()` raises on an *explicit* unavailable
  device; only `auto` walks the ladder. → `test_unavailable_device_raises_rather_than_falling_back`
- **COCO `xywh` vs. internal `xyxy`.** The most common box bug in detection code. The conversion
  happens exactly once, at the loader boundary, and `Box.__post_init__` rejects any degenerate
  result — so a format mix-up surfaces as a loud error rather than a quietly wrong mAP. This matters
  most at specs 02–04, where a box-format slip would corrupt the headline number.
  → `test_degenerate_box_is_rejected`
- **A dataset that loads but is quietly wrong.** Unknown `category_id` and missing image files raise
  rather than being skipped — a silently dropped image would shrink the eval split without telling
  anyone. → `test_unknown_category_is_rejected`, `test_missing_image_is_rejected`
- **The fixture being mistaken for real data.** Guarded three ways: the images are watermarked
  "NOT REAL DATA"; `info.description` in `annotations.json` reads *"Synthetic mini-BDD fixture. NOT
  real BDD100K … never report a metric computed on this (H7)"*; and a test asserts that label
  exists. → `test_fixture_is_labelled_as_synthetic`

## 6 · Honesty contract — H-items at risk, and where

- **H7 — the live one.** The fixture must remain unmistakably synthetic. The risk is not *in* spec
  00; it is that a later spec reports a number computed on `tests/fixtures/`. A fixture mAP is
  meaningless — we drew the boxes. The three guards above exist to make that mistake loud.
- **H8** — the scaffold produces no numbers, so it can claim none. Its contribution is *structural*:
  `run_dir()` establishes `runs/<spec>/<ts>/metrics.json` as the only place a result may come from,
  so no metric can exist only in prose.
- **H6** — encoded in the config shape: `detector` and `vlm` are separate sections for separate
  models, and `cli.vqa`'s docstring states outright that it is a different model from `detect`.
- **H1 / H3 / H4 / H5** — not reachable from spec 00: no model code, no export, no sensors, no
  planner.

## 7 · What spec 01 inherits

The starting point for `/plan 01-data-bdd100k`.

- **Reuse, do not reinvent.** `gdp.config.load_config`, `gdp.data.load_dataset`,
  `gdp.paths.resolve` / `run_dir`, `gdp.seed.set_seed`. There must be exactly one config system and
  exactly one dataset loader.
- **The contract to satisfy.** `configs/bdd100k.yaml` already declares where real BDD100K must land
  (`data/bdd100k/images/100k`, `data/bdd100k/labels/det_20/det_val_coco.json`). Spec 01 changes the
  *source*, not the contract: `load_dataset()` must keep working unchanged, which is precisely what
  keeps every later spec testable offline.
- **The CLI work.** Implement `gdp data`, and remove its row from
  `test_unimplemented_commands_fail_loudly_and_name_their_spec`.
- **The trap spec 01 owns.** The class ↔ prompt-span mapping. `BDD100K_CLASSES` order is
  load-bearing, and `"traffic light"` / `"traffic sign"` are multi-token. Grounding-DINO scores
  **token spans, not class labels**, so mapping a class to its span in
  `"pedestrian. rider. car. …"` is where spec 01's real difficulty lives — not in the file I/O.

## 8 · Verification

Spec 00's acceptance path, re-run 2026-07-15 to confirm this document describes reality:

```bash
uv run pytest                          # 34 passed
uv run ruff check .                    # All checks passed!
uv run gdp info -c configs/default.yaml  # valid JSON, "device": "mps"
bash scripts/smoke.sh                  # SMOKE OK, exit 0
```

Full path from a clean checkout: `make install && make test && make lint && make smoke`.
