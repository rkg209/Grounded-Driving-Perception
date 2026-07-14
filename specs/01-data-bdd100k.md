# Spec 01 — BDD100K data pipeline

**Status:** done · **Compute:** laptop (conversion) + cluster (storage) · **Depends on:** 00

## Objective

Turn raw BDD100K into the canonical COCO-style contract that spec 00's loader already speaks, and
build the **class-index → prompt-token-span mapping** that Grounding-DINO's contrastive text head
requires. This mapping is the single subtlest piece of Stage-1 data code; getting it wrong
produces a model that trains without error and detects nothing.

## Inputs / outputs

- **In:** BDD100K images (`100k/train`, `100k/val`) + detection labels (`det_20`). Requires
  registration at bdd-data.berkeley.edu — **not redistributable, never committed** (`.gitignore`).
- **Out:** `data/bdd100k/labels/det_20/det_{train,val}_coco.json` in the same schema as
  `tests/fixtures/mini_bdd/annotations.json`; loadable by the existing `gdp.data.load_dataset`
  with no changes to it.

## Approach

1. **Download + verify.** Document the exact registration steps and record file checksums. Data
   lives on the cluster; the laptop keeps only the fixture.
2. **Convert** BDD's native JSON → COCO xywh, using the 10-class taxonomy in
   `gdp.config.BDD100K_CLASSES` (order is load-bearing — it defines class indices).
3. **Prompt-span mapping** (`gdp/data/prompt.py`): the prompt is `"pedestrian. rider. car. ..."`.
   Grounding-DINO scores each query token against each box, so a class label must be expressed as
   the **token span** of that class name in the tokenized prompt. Build
   `class_idx → (tok_start, tok_end)` using the tokenizer's offset mapping, **never** by counting
   words — "traffic light" is two words and multiple word-pieces.
4. **Sanity report:** per-class box counts, image counts, degenerate/out-of-frame boxes dropped
   (with the count logged, not silently swallowed).
5. Extend `gdp data` CLI: `gdp data prepare --dataset bdd100k`.

## Acceptance criteria

1. `gdp data prepare` produces `det_val_coco.json` that `load_dataset` reads without modification.
2. A test asserts fixture-parity: the fixture and real BDD JSON pass the *same* validation.
3. Prompt-span test: for each of the 10 classes, decoding the tokens in its span returns the class
   name — including the multi-word `traffic light` / `traffic sign`.
4. Class-count report is written to `runs/01-data/<ts>/stats.json`; counts are non-zero for all 10.
5. Dropped-box count is reported explicitly.
6. All of the above are exercised on the fixture in CI-free laptop tests.

## Honesty contract

- **H7** — report the real class imbalance (BDD is dominated by `car`; `train` is near-absent).
  Do not quietly drop rare classes to flatter later mAP. If a class is too rare to evaluate, say so
  in the results table rather than omitting it.
- **H8** — the val split defined here is the split **all** later mAP numbers must use. Fixing it now
  is what makes zero-shot and fine-tuned comparable.

## Out of scope

No model, no inference, no training. No BDD segmentation/drivable-area labels (detection only).

## Risks

Registration may take days — this blocks 02/03 but **not** 04's design or 05's export work, both of
which can proceed on the fixture and on public COCO images.

## Tasks

Derived from `.claude/plans/01-data-bdd100k.md` §8. One `/implement` per task, `/verify` between each.
All tasks run on the **laptop**, fixture-only — no dataset download is required for any of them.

- [x] **1. `transformers` dep + `gdp/data/` package refactor.**
  `data.py` → `gdp/data/core.py`; `gdp/data/__init__.py` re-exports `Box`, `Sample`, `Dataset`,
  `load_dataset` so `from gdp.data import load_dataset` keeps working verbatim. Add `transformers` to
  `dependencies` in `pyproject.toml`.
  Verify: `uv sync && uv run pytest` — spec 00's existing test suite (34 tests) stays green with zero
  changes to those test files.

- [x] **2. `prompt.py` + golden file + `tests/test_prompt.py`.**
  Implement `PromptSpans` (constructive char-offset cursor, tokenizer `return_offsets_mapping=True`,
  construction-time invariant asserts, L1-normalized `positive_map`). Commit
  `tests/fixtures/prompt_spans_golden.json`. This is the spec's real risk and is done before the
  converter so everything downstream depends on a tested mapping.
  Verify: `uv run pytest tests/test_prompt.py -v` — round-trip decode test passes for all 10 classes
  including `traffic light` / `traffic sign`; invariant-violation cases raise. Test skips (loudly) if
  the tokenizer isn't cached, never silently passes.

- [x] **3. Fixture extension: all 10 classes + raw + dirty fixtures.**
  Extend `scripts/make_fixtures.py` to draw all 10 `BDD100K_CLASSES` across the 4 existing scenes;
  emit `tests/fixtures/mini_bdd_raw/det_val.json` (same boxes, raw BDD format) and
  `tests/fixtures/mini_bdd_raw/det_dirty.json` (one instance each of: `labels: null`, unknown category
  `trailer`, degenerate box, fully-out-of-frame box, edge-straddling box, no-`box2d` label). Keep the
  "NOT REAL DATA" watermark and description (H7).
  Verify: `make fixtures && git diff --stat tests/fixtures/` is deterministic (empty diff on rerun);
  spec 00's tests still assert 4 images and stay green.

- [x] **4. `bdd100k.py` converter + drop accounting + `tests/test_bdd_convert.py`.**
  Raw BDD `det_20` JSON → COCO xywh via `BDD100K_CLASSES` index order. Drop-and-count by reason
  (`unknown_category`, `degenerate`, `out_of_frame`, `no_box2d`); clip-and-count edge-straddling boxes
  separately. Empty-box frames keep their image entry. Deterministic ids (`image_id` = index into
  `sorted(names)`, `annotation_id` = running counter).
  Verify: `uv run pytest tests/test_bdd_convert.py -v` — `test_converted_json_loads_via_spec00_loader`
  (acceptance 1), parity test converting the raw fixture reproduces the committed `annotations.json`
  (acceptance 2), `test_dirty_frames_are_counted_by_reason` asserts exact per-reason drop counts
  (acceptance 5).

- [x] **5. `stats.py` sanity report.**
  Per-class box counts, image counts, dropped/clipped counts, `rare_classes` (below
  `min_boxes_for_eval`), source file + sha256, git sha, timestamp → `runs/01-data/<ts>/stats.json` via
  `gdp.paths.run_dir`.
  Verify: `test_stats_reports_all_ten_classes` on the extended fixture — all 10 classes non-zero
  (acceptance 4); manually inspect one written `stats.json` to confirm it is framed as statistics, not
  a metric (H9).

- [x] **6. CLI `gdp data prepare` + config keys + remove `PENDING` row.**
  `data` becomes a Typer sub-app: `gdp data prepare --dataset {bdd100k,mini_bdd} --split
  {train,val,both}`. `DatasetConfig` gains `raw_labels` and `min_boxes_for_eval: int = 100`;
  `configs/bdd100k.yaml` gains the matching keys in the same task (unknown keys raise by design).
  Remove `"data"` from `PENDING` in `tests/test_cli.py`.
  Verify: `uv run gdp data prepare -c configs/default.yaml --dataset mini_bdd` exits 0 and writes
  `stats.json`; `uv run pytest tests/test_cli.py -v` — the `data` row no longer skipped, passes.

- [x] **7. `docs/bdd100k-download.md`.**
  Registration steps, expected directory tree, checksum manifest format matching what `bdd100k.py`
  verifies against.
  Verify: doc review only — no code path to test; cross-check every path/filename it names against
  what the converter (`task 4`) actually expects.

- [x] **8. Final verify + narrate.**
  Full `/verify` pass, `progress_report.md` entry for the spec as a whole, `specs/README.md` status →
  `done`.
  Verify: `uv sync && uv run pytest && uv run ruff check . && make fixtures &&
  git diff --stat tests/fixtures/ && uv run gdp data prepare -c configs/default.yaml --dataset
  mini_bdd && bash scripts/smoke.sh` — all exit 0.
