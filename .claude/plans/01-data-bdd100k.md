# Plan — Spec 01: BDD100K data pipeline

**Spec:** [`specs/01-data-bdd100k.md`](../../specs/01-data-bdd100k.md) · **Status:** planned ·
**Compute:** laptop (conversion, tests) + cluster (storage of the real dataset) · **Depends on:** 00

> Written *before* implementation, from `specs/01-data-bdd100k.md`. Three design forks were settled
> with the owner up front and are recorded here: the tokenizer/golden-file strategy (§3), extending
> the fixture to all 10 classes (§5), and building the `positive_map` in 01 rather than 03 (§3).
> No code has been written yet. Task order is §8; each task is one `/implement`.

---

## 1 · Context

Spec 00 built the contract: `gdp.data.load_dataset` reads a COCO-style JSON, and
`configs/bdd100k.yaml` already declares where real BDD100K must land. Nothing produces that JSON yet.
Spec 01 changes the **source** (raw BDD100K → our COCO schema) without changing the **contract**, so
every later spec stays verifiable offline on the fixture.

The file I/O is the easy half. The real work is the **class-index → prompt-token-span mapping**.
Grounding-DINO has no class head: it scores each box against the *text tokens* of the prompt
`"pedestrian. rider. car. …"`. A class is therefore a **token span**, not a label. Get this wrong and
the model trains without error, runs without error, and detects nothing — or worse, reports a
plausible-but-wrong mAP. This mapping is what specs 02 (span → class at eval) and 03 (`class_labels`
→ the contrastive loss) are both built on, so spec 01 owns it and must test it to destruction.

Two things found during exploration, which the plan resolves:

- **`transformers` is not a dependency and there is no HF cache.** The span mapping needs a real
  tokenizer (Grounding-DINO's is `bert-base-uncased`).
- **The fixture only draws 4 of the 10 classes** (`rider`, `truck`, `train`, `motorcycle`, `bicycle`,
  `traffic sign` have zero boxes), so acceptance criterion 4 ("counts non-zero for all 10") cannot
  currently be exercised offline as criterion 6 demands.

## 2 · What gets built

| File | Responsibility |
|---|---|
| `src/gdp/data/__init__.py` | Re-exports `Box`, `Sample`, `Dataset`, `load_dataset`. **The spec-00 import path `from gdp.data import load_dataset` must keep working verbatim.** |
| `src/gdp/data/core.py` | Today's `src/gdp/data.py`, moved unchanged. |
| `src/gdp/data/prompt.py` | `PromptSpans` — the token-span mapping. The heart of this spec. |
| `src/gdp/data/bdd100k.py` | Raw BDD `det_20` JSON → our COCO schema, with drop accounting. |
| `src/gdp/data/stats.py` | The sanity report → `runs/01-data/<ts>/stats.json`. |
| `src/gdp/cli.py` | `data` becomes a sub-app: `gdp data prepare --dataset {bdd100k,mini_bdd}`. |
| `src/gdp/config.py` | `DatasetConfig` gains `raw_labels` + `min_boxes_for_eval`. |
| `scripts/make_fixtures.py` | Draw all 10 classes; also emit the **raw-BDD-format** fixtures. |
| `docs/bdd100k-download.md` | Registration steps, expected tree, checksum manifest. |
| `tests/test_prompt.py`, `tests/test_bdd_convert.py` | New. |

Reuse, do not reinvent: `gdp.config.load_config`, `gdp.paths.resolve` / `run_dir`,
`gdp.data.load_dataset`, `gdp.logging.get_logger`. One config system, one loader.

`data.py` becomes a package because the spec names `gdp/data/prompt.py`; the `__init__` re-export is
what keeps that refactor invisible to spec 00's tests.

## 3 · The token-span mapping (`prompt.py`)

```python
@dataclass(frozen=True)
class PromptSpans:
    prompt: str                                # "pedestrian. rider. car. ... traffic sign."
    classes: tuple[str, ...]
    input_ids: tuple[int, ...]
    char_spans: tuple[tuple[int, int], ...]    # per class index
    token_spans: tuple[tuple[int, int], ...]   # per class index, [start, end)

    def token_class_ids(self) -> np.ndarray:   # len(input_ids); -1 where not a class token
    def positive_map(self, max_text_len: int = 256) -> np.ndarray:  # (num_classes, max_text_len)
    def to_json(self) -> dict: ...
```

Construction rules — each one is a bug that has bitten real Grounding-DINO ports:

1. **Build the char offsets constructively**, with a cursor, while joining the classes into the
   prompt. Never `str.find(class_name)`: `"traffic light"` and `"traffic sign"` share the prefix
   `"traffic"`, and a substring search is a coin flip waiting to happen.
2. **Get token spans from the tokenizer's `return_offsets_mapping=True`**, never by counting words.
   `"traffic light"` is two words and several word-pieces. A token belongs to class *i* iff its char
   offset overlaps class *i*'s char span; special tokens (offset `(0, 0)`) belong to nothing.
3. **Assert the invariants at construction**: every class owns ≥ 1 token; spans are disjoint and
   strictly increasing; no class span is empty. A silent empty span is the "trains fine, detects
   nothing" failure.
4. **`positive_map` rows are L1-normalized** (the MDETR/Grounding-DINO convention the loss expects),
   and `max_text_len` must match what the processor pads to.

**Tokenizer + golden file (the agreed approach).** Add `transformers` to `dependencies` (spec 02
needs it regardless). Tests load the tokenizer from the HF cache and compare the computed spans
against a committed `tests/fixtures/prompt_spans_golden.json`. The golden file is the point: a
tokenizer-version bump that silently shifts a span would otherwise change every training label in
spec 03 without a single test going red. If the tokenizer is neither cached nor downloadable, the
test **skips with a loud reason** rather than failing — and `make tokenizer` pre-caches it (~1 MB).

**Round-trip test (acceptance 3):** for all 10 classes, `decode(input_ids[start:end]).strip()` returns
the class name — with `traffic light` and `traffic sign` as the ones that actually matter.

## 4 · The converter (`bdd100k.py`)

Raw `det_20` frames look like
`{"name": "…jpg", "labels": [{"category": "car", "box2d": {"x1","y1","x2","y2"}}, …]}`.

- `box2d` is absolute **xyxy** → our COCO output is absolute **xywh**. Convert once, at this
  boundary, and let `Box.__post_init__` (spec 00) reject anything degenerate downstream.
- Category names map to `BDD100K_CLASSES` **by index** — that order is load-bearing, and the emitted
  `categories[].id` must be `0..9` to match the fixture schema exactly.
- **Drop accounting, never silent** — every dropped box is counted **by reason** and reported:
  `unknown_category` (det_20 ships ignore-classes such as *other person* / *other vehicle* /
  *trailer* beside the official 10 — verify against the real file on download; the converter counts
  and drops them rather than crashing), `degenerate` (zero/negative area), `out_of_frame` (entirely
  outside the image), `no_box2d` (a label with attributes only). Boxes that merely *straddle* the
  edge are **clipped and counted as clipped, not dropped**.
- **A frame with `labels: null` or zero surviving boxes keeps its image entry, with no boxes.**
  Dropping empty images would silently shrink the val split — and this split is the one every later
  mAP is computed on (H8).
- Image size: BDD100K 100k is uniformly 1280×720. Take it from config, and **verify it against a
  sample of N real images**, erroring on mismatch rather than trusting the assumption.
- Deterministic ids: `image_id` = index into `sorted(names)`; `annotation_id` = running counter. Same
  input bytes → same output bytes.

## 5 · The fixture, extended (this is what makes 01 testable offline)

`scripts/make_fixtures.py` is extended to:

1. **Draw all 10 classes** across the same 4 scenes, so "non-zero for all 10" (acceptance 4) is a real
   assertion on the laptop. Spec 00's tests assert *4 images*, not a box count, so they still pass.
   The "NOT REAL DATA" watermark and the `NOT real BDD100K` description stay (H7).
2. **Also emit the same boxes in raw BDD format** → `tests/fixtures/mini_bdd_raw/det_val.json`. The
   converter then has an offline input, and the parity test (acceptance 2) has real teeth:
   *convert(raw fixture) must reproduce the committed `annotations.json`*, which was written by an
   independent code path.
3. **Emit a deliberately dirty raw file** → `tests/fixtures/mini_bdd_raw/det_dirty.json`, containing
   one of each nasty case: `labels: null`, an unknown category (`trailer`), a degenerate box, a box
   fully outside the frame, a box straddling the edge, a label with no `box2d`. The test asserts the
   **exact drop counts per reason** (acceptance 5).

## 6 · The sanity report (`stats.py`)

`gdp data prepare` writes `runs/01-data/<ts>/stats.json` via `paths.run_dir`:

```json
{"split": "val", "images": 10000, "images_without_boxes": 137, "boxes": 185526,
 "per_class": {"car": 102506, "...": 0}, "dropped": {"unknown_category": 4113, "degenerate": 2,
 "out_of_frame": 0, "no_box2d": 0}, "clipped": 918,
 "rare_classes": ["train"], "min_boxes_for_eval": 100,
 "source": {"file": "det_val.json", "sha256": "…"}, "git_sha": "…", "created": "…"}
```

**H7 lives here.** BDD is dominated by `car`; `train` is nearly absent and may well be *zero* in val.
So: the all-10-non-zero assertion is a **fixture** test, while on real data a class below
`min_boxes_for_eval` is **flagged** in `rare_classes` — surfaced as "too rare to evaluate" in the
results table rather than quietly dropped to flatter later mAP. And `stats.json` is *dataset
statistics, not a metric* — it must never be presented as a result (H9).

## 7 · CLI and config

- `data` becomes a Typer sub-app: `gdp data prepare --dataset {bdd100k,mini_bdd} --split {train,val,both}`.
  `--dataset mini_bdd` runs the *same* code path on the fixture — that is the offline smoke path.
- **Remove `"data"` from `PENDING` in `tests/test_cli.py`.** That parametrized test is the ratchet:
  the row leaves exactly when the command starts working.
- `DatasetConfig` gains `raw_labels` (raw BDD JSON dir) and `min_boxes_for_eval: int = 100`;
  `configs/bdd100k.yaml` gains the matching keys. Both must land in the *same* task — unknown config
  keys raise by design.

## 8 · Task order (one `/implement` each, `/verify` between)

1. `transformers` dep + `gdp/data/` package refactor (re-export; spec-00 tests must stay green).
2. `prompt.py` + golden file + `tests/test_prompt.py`. **Do this before the converter** — it is the
   spec's real risk, and everything downstream depends on it.
3. Fixture extension: all 10 classes + raw + dirty fixtures (`make fixtures`).
4. `bdd100k.py` converter + drop accounting + `tests/test_bdd_convert.py`.
5. `stats.py` + `runs/01-data/<ts>/stats.json`.
6. CLI `gdp data prepare` + config keys + remove the `PENDING` row.
7. `docs/bdd100k-download.md` + checksum manifest.
8. `/verify`, `progress_report.md` entry, `specs/README.md` → done.

## 9 · Honesty contract

- **H7** — real class imbalance is reported, not smoothed. Dropped boxes are counted by reason and
  printed. Rare classes are flagged, never omitted.
- **H8** — the val split defined here is *the* split for every later mAP. The checksum manifest is
  what makes "the same split" checkable rather than asserted.
- **H9** — `stats.json` is dataset statistics. It is not a metric and must not be reported as one.
- **H1 / H3 / H4 / H5** — not reachable: no model, no inference, no training, no sensors.

## 10 · Verification

Offline, on the M4 — the whole spec is verifiable with no dataset:

```bash
uv sync                                              # transformers resolves
uv run pytest                                        # spec-00's 34 tests still green + new ones
uv run ruff check .
make fixtures && git diff --stat tests/fixtures/     # regeneration is deterministic
uv run gdp data prepare -c configs/default.yaml --dataset mini_bdd
uv run python -c "from gdp.data import load_dataset; print(len(load_dataset(...)))"  # converted JSON loads unchanged
bash scripts/smoke.sh                                # exits 0
```

Each acceptance criterion, and what discharges it:

| # | Criterion | Discharged by |
|---|---|---|
| 1 | `prepare` output is read by `load_dataset` unmodified | `test_bdd_convert.py::test_converted_json_loads_via_spec00_loader` |
| 2 | Fixture and real BDD pass the *same* validation | the converter has one code path; parity test vs. committed `annotations.json` |
| 3 | Each class's span decodes to its class name | `test_prompt.py::test_span_round_trips` (all 10, incl. both two-word classes) |
| 4 | `stats.json` written; all 10 counts non-zero | `test_stats_reports_all_ten_classes` on the extended fixture |
| 5 | Dropped-box count reported explicitly | `test_dirty_frames_are_counted_by_reason` (exact counts) |
| 6 | All of it exercised on the fixture | every test above is fixture-only; no download needed |

On the real data (user runs on the cluster, after registration):

```bash
uv run gdp data prepare -c configs/default.yaml -c configs/bdd100k.yaml --dataset bdd100k --split both
```

Then eyeball `runs/01-data/<ts>/stats.json` — 10k val images, `car` dominant, `train` near-zero. **If
a class is unexpectedly zero, the taxonomy mapping is wrong, not the dataset.**

## 11 · Risks

- **Registration may take days.** It blocks 02/03 but nothing else: every task above is done and
  tested on the fixture, so spec 01 can be *complete and merged* before the data arrives. The only
  step that waits is running `prepare` on the real file.
- **Tokenizer drift** silently shifting spans → the committed golden file is exactly the guard.
- **det_20's extra ignore-classes** — the converter counts-and-drops unknown categories, so an
  unexpected category name is a number in the report, not a crash and not a silent swallow.
