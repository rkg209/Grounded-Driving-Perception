# Remaining Action Plan — from "no measured results" to a defensible resume claim

**Created:** 2026-09-03 · **Companion to:** [`PROJECT_RESULTS_AUDIT.md`](PROJECT_RESULTS_AUDIT.md)
**Repo state this plan assumes:** `main` @ `f756beb`, all specs' laptop-side code implemented and
passing, `data/` empty of real datasets, zero cluster jobs ever submitted.

> **Read this first.** The audit's finding is that the *instrument* is finished and no
> *measurement* has been taken. Therefore **nothing in this plan is a code improvement**. Almost
> every item is "download a dataset", "submit a job that already exists", or "author the input the
> tooling is waiting for". Two items are code fixes, and both exist only because they invalidate a
> number the project intends to publish.
>
> **The critical path is dataset access.** Actions 1 and 5 are registrations with human approval
> latency measured in days. Start both **today**, in parallel, before anything else in this file.
>
> **Rules that still bind every action below:** `CLAUDE.md` §4 (never train in-session on the M4 —
> all cluster jobs below are submitted **by the user, on Rudra**, never from a Claude session; the
> `guard-bash` hook enforces this and blocked an earlier attempt to write this very file), §5
> (append a `progress_report.md` entry after every change), §7 (no `Co-Authored-By` trailer in
> commits), and the Honesty Register H1–H9 — in particular H8: **no metric without its paired
> baseline on the same split.**

---

## Priority summary

| # | Action | Blocks | Effort | Impact |
|---|---|---|---|---|
| 1 | Register + download **BDD100K** on PARAM Rudra | M1, M2, M3, M7, M8 | Hours of work, **days of approval latency** | 🔴 Critical path |
| 2 | Run spec 02 — **zero-shot mAP baseline** | M2 (H8 baseline) | 1 job, ~4 h | 🔴 Critical |
| 3 | Run spec 03 — **detector fine-tuning** → mAP delta | **The Stage-1 headline** | 1 job, ≤24 h | 🔴 Critical |
| 4 | Fix the **latency benchmark device asymmetry**, then re-run spec 05 on real BDD100K val | M7, M8, M9 | ~2 h code + ~1 h run | 🔴 High |
| 5 | Register + download **nuScenes v1.0-trainval + DriveLM** | M4, M5, M6 | Hours of work, **days of latency** | 🟠 High (start with #1) |
| 6 | Author the **150–300 grounding phrases**, freeze, run spec 04 | M3 | 6–10 h of human authoring | 🟠 High |
| 7 | Run specs 06→07→08 — **DriveLM chain** → VQA delta | **The Stage-2 headline** | ~40 h wall-clock of jobs | 🟠 High |
| 8 | Install `openai` + re-verify the **official-scorer round-trip** | Validity of M4/M5 | 5 min | 🟡 Medium (do before #7) |
| 9 | Re-run spec 09 demo with **real checkpoint + adapter**, record video | Portfolio artifact | ~2 h | 🟡 Medium |
| 10 | `gdp report snapshot && render`, `/claims-check`, rewrite resume bullets | Everything | ~1 h | 🟡 Final gate |

**Dependency chain:** `1 → 2 → 3 → {4, 6}` and `5 → 7` (with `8` before `7`), then `9`, then `10`.
Actions 1 and 5 run in parallel from day zero. Action 8 can be done right now, on the laptop, in
five minutes.

---

## Action 1 — Register and download real BDD100K on PARAM Rudra

> **✅ DONE 2026-09-22** — see `progress_report.md` [SEQ-0122]. Not via the Berkeley portal (403
> Forbidden) but the Kaggle mirror `awsaf49/bdd100k-dataset`, verified against the official class
> histogram and split sizes before conversion. On Rudra at
> `/scratch/IITB/ai-at-ieor/24m1531/4_Honda/data/bdd100k/`; converted to
> `runs/01-data/20260922-225014/` — val 10,000 images / 185,945 boxes, train 69,863 / 1,273,707.
> The approval latency this action warned about did not materialise; total elapsed was ~1 hour.
> **Action 2 is unblocked.** The rest of this section is kept as written, for the record.

**Action.** Complete the BDD100K account registration and terms acceptance, then download the
`100k` images archive and the `det_20` detection labels onto the cluster's scratch storage. Follow
`docs/bdd100k-download.md` for the exact URLs, expected layout, and checksums, and
`PARAM_Rudra_GDP_Cluster_Guide.md` §2 for where on Rudra to put them. Then point `configs/bdd100k.yaml`
at the real paths and run `uv run gdp data prepare -c configs/default.yaml -c configs/bdd100k.yaml
--dataset bdd100k --split val` (and `--split train`) to produce the COCO-style conversion with its
drop accounting.

**Why.** The audit's confirmed cause #1: `data/` is 88 KB and contains no real dataset. This single
absence blocks metrics M1, M2, M3, M7 and M8 — including **both** Stage-1 headline numbers. Nothing
downstream can start.

**Current problem.** Specs 01–05 have run only against `tests/fixtures/mini_bdd/`: 4 synthetic
images, 10 boxes. Every artifact they produced carries `is_synthetic: true` and renders as
"Not yet measured".

**Expected outcome.** Real BDD100K on cluster storage and a real
`runs/01-data/<ts>/det_val_coco.json` + `stats.json` reporting on the order of 10k val images and
~185k boxes (the published BDD100K val figures — **verify against your own `stats.json`, do not
quote them until you have**). Registration approval time is outside your control and this is not
guaranteed to complete on any particular day.

**How to execute.**
1. Read `docs/bdd100k-download.md` end to end before starting — it names the exact archives.
2. Register at the BDD100K portal, accept terms, obtain download links.
3. On Rudra (login node is fine for transfer, never for compute):
   `cd /scratch/24m1531/4_Honda && mkdir -p data/bdd100k` then download images + `det_20` labels there.
4. Verify the layout matches what `configs/bdd100k.yaml` expects (`dataset.root`,
   `dataset.annotations`, `dataset.raw_labels`).
5. Run the converter for `val` first, then `train`.

**How to validate.** `runs/01-data/<ts>/stats.json` exists with `images` in the thousands,
`boxes` in the hundreds of thousands, `dropped` counts all accounted for, and **no `is_synthetic`
flag set true anywhere downstream**. Cross-check `per_class` against BDD100K's published class
distribution — a wildly skewed histogram means a broken conversion, not a hard dataset.

**Metrics affected.** M1, M2, M3, M7, M8 (enables all of them; measures none).

**Resume impact.** None directly — this is the precondition for every Stage-1 claim.

**Dependencies.** Cluster access (`24m1531@paramrudra.iitb.ac.in`), scratch quota for a multi-GB
dataset. **Start this today.**

---

## Action 2 — Run spec 02: the zero-shot mAP baseline

> **✅ DONE 2026-09-23** — see `progress_report.md` [SEQ-0126]. Job 409116 (A100, 29 min, commit
> `1a8456e`): **mAP 0.2246 / mAP50 0.3841** on the full 10,000-image val split, `is_synthetic:
> false`, `map_score_floor` 0.05, `box_threshold` 0.25 `chosen_on: train` from a 2,000-image
> sweep. Artifact `runs/02-zeroshot/20260923-004055/metrics.json` on Rudra scratch **and pulled to
> the laptop** ([SEQ-0127]) — this is the
> `ZEROSHOT_METRICS` input for Action 3. The notes below are kept for the record.

> **⚠️ PARTIAL 2026-09-22** — see `progress_report.md` [SEQ-0124]. Job 409062 (A100, ~23 min,
> commit `3251423`) measured **mAP 0.1983 / mAP50 0.3253 on the full 10,000-image real val split**,
> `is_synthetic: false`, threshold 0.25 `chosen_on: train`. **But** it was submitted as a plumbing
> test with `SWEEP_LIMIT=20`, so the threshold was swept on 20 train images instead of 2,000. No
> val leakage, so the number is honest — but it is **provisional until the sweep is re-run at
> `SWEEP_LIMIT=2000`** (clear `runs/02-zeroshot/slurm-state/` first, or the resume cache skips it)
> and val is re-scored at whatever threshold that picks. The artifact is on Rudra scratch only;
> the laptop has not pulled it, so the README still renders "Not yet measured" — correctly.
>
> **🔧 Fixed before the re-run, 2026-09-23** — see [SEQ-0125]. Job 409062 also detected val *at*
> the 0.25 threshold, so its mAP was computed over a truncated precision-recall curve (understated,
> not comparable to published numbers). `zeroshot_eval.slurm` now detects val at a 0.05 score floor
> and applies the threshold to precision/recall only; `metrics.json` records `map_score_floor`.
> Re-run from commit ≥ this fix with `rm -rf runs/02-zeroshot/slurm-state` and the default
> `SWEEP_LIMIT=2000`. **0.1983 is superseded, not just provisional.** Expect the val predictions
> file to be much larger (~140 boxes/image at 0.05 on the train slice → ~1.4M boxes on val).

**Action.** Submit `scripts/slurm/zeroshot_eval.slurm` on Rudra (from your own shell, not a Claude
session), against real BDD100K val.

**Why.** H8 is absolute: the fine-tuned mAP is **meaningless without its zero-shot baseline on the
same split**. `specs/README.md` calls this "the bar for H8". Action 3 cannot produce a publishable
number until this one exists.

**Current problem.** `runs/02-zeroshot/20260726-214630/metrics.json` reports `map: 0.0` on 4
synthetic images with `tp: 0, fp: 4, fn: 10` — a fixture rehearsal, correctly stamped
`is_synthetic: true`.

**Expected outcome.** `runs/02-zeroshot/<ts>/metrics.json` with a real `map`, `map50`, per-class AP,
and the operating threshold the sweep chose. Zero-shot Grounding-DINO-tiny on driving classes is
typically modest — **whatever it is, that is the baseline; do not tune it upward to look good, its
job is to be honest.**

**How to execute.**
1. Confirm Action 1's converted val split exists and the config points at it.
2. Read `scripts/slurm/zeroshot_eval.slurm` and check its `--time` (4 h requested), partition and
   GPU request are valid for your Rudra allocation.
3. Submit it; monitor with `squeue -u 24m1531`.
4. On completion record the **actual** runtime from `sacct` — `docs/reproduction.md` explicitly asks
   for measured runtimes to replace the `#SBATCH --time` requests.

**How to validate.** `metrics.json` has `is_synthetic: false` (or absent), `num_images` in the
thousands, `predictions_meta.json` lists the images actually processed (the SEQ-0043 `--limit` bias
guard), and `map > 0`. A `map` of exactly 0.0 on real data means a prompt/label-mapping bug — check
the class↔prompt-span map before believing it.

**Metrics affected.** **M1** (measures it); M2 (enables it).

**Resume impact.** Enables the honest phrasing "zero-shot X → fine-tuned Y". Alone, a baseline is
not a claim.

**Dependencies.** Action 1.

---

## Action 3 — Run spec 03: detector fine-tuning → the Stage-1 headline

> **🔧 Fixed before first submission, 2026-09-23** — see `progress_report.md` [SEQ-0130]. Three
> bugs would have wasted the allocation: (1) `--resume` restored step/optimizer but loaded the
> *pretrained* weights, silently discarding all training before a walltime kill; (2) the job's
> resume glob picked up the overfit gate's `checkpoint-200` on the very first submission;
> (3) ~420 × ~2 GB checkpoints (~800 GB) with no retention. Now: weights load from the checkpoint,
> only post-gate checkpoints are resume candidates, `keep_last_checkpoints: 3`, `num_workers: 6`,
> log records carry `time`, walltime 72 h (≈209k steps). **Submit from commit `237bbd7` or later.**
>
> **❌ Job 409215 failed the gate again (3.08); 1,000-step `srun` 409220 plateaued at ~3.1**
> — [SEQ-0134]/[SEQ-0135]. Cause: HF computes the encoder box losses on `.detach()`ed boxes, so
> ~2.5 of the gate's total had zero gradient. Fixed in `237bbd7` ([SEQ-0136]): all encoder terms
> scaled out (`enc_loss_scale: 0`), target still 1.0. Decoder-only loss on those runs: 0.78 @200,
> 0.66 @1000.
>
> **✅ M2 MEASURED 2026-09-24**, [SEQ-0145]. Job 409265 COMPLETED (1 d 11 h 52 m, commit
> `237bbd7`): **BDD100K val mAP 0.2246 → 0.3003 (+0.0758); mAP50 0.3841 → 0.5228**, 10,000
> images, same threshold/floor, **no per-class regression**. Probe (demo): the fine-tuned model
> labels ordinary cars "police car" ~4.9× more often. **✅ Action 3 DONE:** acceptance-5 crops
> generated ([SEQ-0146]; judged at the 0.05 floor, label accordingly); spec 03 → done.
> Do not quote P/R at 0.25 as a precision drop (see [SEQ-0145]).
>
> **🏃 RUNNING — job 409265 (submitted 2026-09-23)**, [SEQ-0137]. Gate passed (0.833 < 1.0); full
> run started fresh in `runs/03-finetune/20260923-032728/` at 0.606 s/step, so ~35 h of
> training, then val eval + `gdp compare` + probe run automatically. Next session: check
> `sacct -j 409265`; on completion pull `comparison.json` and close Action 3. If it times out
> or is pre-empted, resubmit the same command; it resumes from the latest post-gate checkpoint.
>
> **❌ Job 409183 (2026-09-23) failed the overfit gate** (final loss 2,783) — [SEQ-0132]. Cause:
> HF's encoder-proposal class loss is ~65,000× every other term on the pretrained checkpoint.
> Fixed in `edaef62` ([SEQ-0133]) by dropping that term (`enc_class_loss_weight: 0`), a
> documented spec-03 deviation. Measured step time: **0.67 s/step → ~39 h** for the full run.
> The same two resume bugs existed in the VLM path (`finetune_vlm.slurm`, `train_vlm`) —
> **fixed in `733bc79`** ([SEQ-0139]), ahead of Action 7.

**Action.** Submit `scripts/slurm/finetune_detector.slurm`. The script runs the full chain: overfit
gate → full fine-tuning → re-score on val at the replayed threshold → `gdp compare` → the
open-vocabulary forgetting probe.

**Why.** This produces **the single most important number in the project** — the zero-shot →
fine-tuned mAP delta. Without it the project has no headline result, and the charter's central claim
("domain-adapted with *measured* gains") is unsupported.

**Current problem.** Fine-tuning has never been executed. `runs/03-finetune/` contains five empty
timestamp directories from fixture rehearsals. `specs/README.md`'s spec-03 note lists exactly this
as blocker 3.

**Expected outcome.** `runs/03-finetune/<ts>/comparison.json` carrying both mAPs and the delta, plus
a checkpoint directory. A positive delta is *expected* but **not guaranteed** — fine-tuning an
open-vocabulary detector can degrade its open-vocabulary behaviour, which is precisely why the
forgetting probe is in the script. **If the delta is negative or the probe shows collapse, that is a
result too, and per H7 it gets reported, not rerun until it looks better.**

**How to execute.**
1. Verify Action 2's `metrics.json` path is wired into the script's `ZEROSHOT_METRICS` input.
2. Inspect `configs/train_detector.yaml`: `lr: 1e-4`, `backbone_lr_mult: 0.1`, `epochs: 12`,
   `batch_size: 4`, `freeze_text_encoder: true`, `max_grad_norm: 0.1`.
   ⚠️ **`grad_accum` must stay `1`** — `src/gdp/train/trainer.py:80` raises `NotImplementedError`
   for any other value rather than silently lying about the effective batch size. If you need a
   larger effective batch on an A100, raise `batch_size`, not `grad_accum`.
3. **The overfit gate is the go/no-go.** It trains on `overfit_images: 20` for up to
   `overfit_max_steps: 200` targeting `overfit_loss_target: 1.0`. If it fails, stop — the recipe is
   wrong and a 24 h run would waste the allocation. Debug the gate first.
4. Submit the job (24 h requested). Checkpoints resume via `save_every: 500`, so a timeout is
   recoverable.
5. Record the real runtime and peak GPU memory from `sacct`.

**How to validate.** `comparison.json` exists with both numbers on the same split at the same
replayed threshold; the training log (`gdp.train.logging_jsonl`) shows a loss curve that actually
decreases; the open-vocab probe output shows what the fine-tune cost in out-of-taxonomy detection
(never a metric — H9 — but essential context).

**Metrics affected.** **M2** (measures it), M1 (pairs with it).

**Resume impact.** **This is the bullet.** "Fine-tuned Grounding-DINO-tiny on BDD100K; improved
detection mAP@[.5:.95] from X to Y on the official val split (N images), and quantified the
open-vocabulary forgetting cost." That is a claim a Honda J6/J7 interviewer will engage with.

**Dependencies.** Actions 1 and 2. Never run on the laptop — `guard-bash` will block it.

---

## Action 4 — Fix the latency benchmark's device asymmetry, then re-run spec 05 on real data

**Action.** Two parts, in order.

*Part A (code, laptop, ~2 h).* Make the three deployment variants comparable and self-describing:
1. In `src/gdp/cli.py:1738` (`deploy bench`) and `:1794` (`deploy evaluate-variants`), stop passing
   `cfg.device` (which resolves to **MPS**) to `GroundingDinoDetector` while `OnnxDetector` runs on
   onnxruntime's **CPU** provider. Either pin the PyTorch variant to CPU for the benchmark, or add
   an explicit `--device` option and run the comparison twice (all-CPU, and all-MPS/GPU where the
   provider supports it). **All-CPU is the honest default for an "edge target" claim** and is the
   minimum fix.
2. Extend `collect_hardware_info` (`src/gdp/deploy/bench.py:126-141`) to record, **per variant**,
   the torch device and the onnxruntime execution provider actually used, and write them into
   `latency.json`.
3. Either make `peak_rss_bytes` genuinely per-variant (it is currently
   `resource.getrusage(RUSAGE_SELF).ru_maxrss` — a process-global high-water mark, which is why all
   three variants report an identical 5,026,185,216 bytes), or rename the field and document it as
   a whole-process figure so it can never be quoted as a per-variant memory result.
4. Add a regression test asserting `latency.json` records a device/provider for every variant.
5. Append a `progress_report.md` entry (What/Why/How/Issues/Verification) — mandatory per `CLAUDE.md` §5.

*Part B (run, laptop, ~1 h).* Re-run the full spec-05 chain against **real BDD100K val**:
`gdp deploy export` → `quantize` → `bench` → `evaluate-variants`.

**Why.** The audit found (§4.2, a **new finding not recorded anywhere in the repo**) that the one set
of real measurements this project owns is invalid as a quantization comparison: fp32-pt ran on MPS,
both ONNX variants on CPU, and `latency.json` does not disclose it. That is why INT8 appears *slower*
(3.14 s) than fp32 PyTorch (1.66 s). Publishing that curve as an edge-deployment result would not
survive one question from a deployment engineer.

**Current problem.** M7 is measured but invalid; M8's curve is plotted from fixture mAPs of
0.0/0.0/0.045 and carries no information; M9 is not per-variant. The benchmark also ran at the
fixture's 360×640 — real BDD100K frames are 720×1280, **4× the pixels**.

**Expected outcome.** A `latency.json` where every variant names its execution target, and a
`curve.png` plotting real-val mAP against p50 at real resolution. **Whether INT8 turns out faster is
not guaranteed** — dynamic quantization of a transformer detector on ARM CPU sometimes is not. A
measured, explained "no speedup" is a legitimate and interesting result; a fabricated speedup is not.

**How to execute.** Part A on the laptop (this is laptop-native by design — the M4 *is* the edge
target, per H3; do not move it to the cluster). Part B needs real BDD100K val images locally, or run
it on Rudra's CPU nodes and label the hardware accordingly — **but note that changing the hardware
changes what "edge target" means**, so prefer bringing a val subset to the M4 and saying exactly how
many images were used.

**How to validate.** `latency.json` has a device/provider per variant and they match; `metrics.json`
has `is_synthetic: false` with a real `num_images`; `curve.png` shows three points with distinguishable
mAP values; a variant's p50 is stable across two runs (interleaving with 50 warmup / 200 timed is
already in place).

**Metrics affected.** **M7, M8, M9.**

**Resume impact.** Enables the J6/J7 "edge device deployment" bullet honestly: "Exported
Grounding-DINO to ONNX with INT8 dynamic quantization; benchmarked p50/p95/p99 and FPS on an Apple
M4 CPU at 720×1280, publishing the accuracy-vs-latency tradeoff." Without Part A this bullet is
indefensible.

**Dependencies.** Part A: none — **can be done today**. Part B: Action 1 (real val images).

---

## Action 5 — Register and download nuScenes v1.0-trainval + DriveLM

> **✅ DONE 2026-09-24** ([SEQ-0149]): `runs/06-data/20260924-160809/` on Rudra (commit
> `67fe482`). Train 592 scenes / 321,893 QA; val = seeded 15% holdout, 104 scenes / 4,284 QA,
> 0 overlap. ⚠️ **For Action 7:** tag-0 answers are near-constant ("No." 97.8%, "Going ahead."
> 92.6%); a constant answer per category scores 0.715. Always report accuracy next to that
> majority-answer rate. Behavior (26.1% majority) is the only informative accuracy category.
>
> **🏃 STARTED 2026-09-23**, in parallel with Action 3's job 409265.
> **Downloads ✅ done + verified** ([SEQ-0140]): `data/drivelm/v1_1_train_nus.json`, 6 × 4,072
> camera images from DriveLM's own `drivelm_nus_imgs_train.zip` (no nuScenes blobs needed),
> `data/nuscenes/v1.0-trainval/scene.json` (850 scenes); 0 of 24,432 referenced images missing.
> **Conversion attempts:** `small` nodes hang at Python start-up ([SEQ-0141]/[SEQ-0142]; use a
> `gpu` node). The first real run kept **0** QA: the real file has no `tag` ([SEQ-0143]). Fixed in
> code by vendoring DriveLM's `extract_data.py` ([SEQ-0144]).
> Run at `e8976e2`: tags work (29,450 official selections), but **val was empty**: DriveLM's
> answered file is 696/696 nuScenes-train ([SEQ-0147]). User decision: local val = seeded 15%
> scene holdout (labelled as such everywhere), plus the official val questions submitted to
> DriveLM's own server for the headline, if it still scores ([SEQ-0148]; spec 06 status note).
> **Remaining:** push → `git pull` on Rudra (expect ≥ the holdout commit) → re-run
> `prepare-drivelm --split both` on a `gpu` node → check `holdout.json`, `stats_*.json`, scene
> overlap.
> Needed: DriveLM `v1_1_train_nus.json`, nuScenes `v1.0-trainval_meta.tgz` (→ `scene.json`),
> and the six `samples/CAM_*` folders (camera-only, H4). No nuScenes devkit is needed: the
> official split lists are vendored in `src/gdp/data/nuscenes_splits.json`.

**Action.** Complete the nuScenes registration (separate from BDD100K) and download
`v1.0-trainval` plus DriveLM's *answered* train file to Rudra scratch, per `docs/drivelm-download.md`.
Then run the conversion: `uv run gdp data prepare-drivelm -c configs/default.yaml -c configs/drivelm.yaml
--dataset drivelm --split both` (CPU-only, no job script needed, but it needs cluster storage).

**Why.** Blocks the entire Stage-2 chain — M4, M5, M6, and therefore the project's *risk-reasoning*
claim, which lives entirely in DriveLM's planning/behavior categories (`specs/README.md`, "Where risk
assessment lives"). This is a **separate registration** from BDD100K with its own approval latency,
which is why it starts on day zero alongside Action 1 rather than after Stage 1 finishes.

**Current problem.** No DriveLM or nuScenes data exists. `runs/06-data/*` came from
`tests/fixtures/mini_drivelm/` — 4 synthetic scenes.

**Expected outcome.** Real `train.jsonl` / `val.jsonl` with `stats_train.json` / `stats_val.json`
recording thousands of QA pairs across the four categories, a scene-clean split (the leakage gate
enforces it), and every record carrying its `tag` field (the scorer-routing field found missing in
spec 08 task 3a — without it the official scorer cannot route items).

**How to execute.**
1. Read `docs/drivelm-download.md` fully, **especially §4** — the split resolution.
2. Register for nuScenes, accept terms, download `v1.0-trainval` metadata + the camera images the
   DriveLM frames reference; download DriveLM's answered train JSON.
3. Point `configs/drivelm.yaml` at the real paths.
4. Run `prepare-drivelm --split both` on a compute node (not the login node).

**How to validate.** `stats_*.json` shows realistic per-category counts, the drop accounting adds up,
the scene-leakage gate passes, and every record has a non-empty `tag` list. **Carry forward the
caveat in every artifact:** this is *not* the DriveLM leaderboard split — the challenge val/test
answers are withheld behind EvalAI, so the repo partitions DriveLM's answered train file by the
official nuScenes 700/150 scene lists. That sentence must appear beside every Stage-2 number.

**Metrics affected.** M4, M5, M6 (enables; measures none).

**Resume impact.** None directly; precondition for the Stage-2 headline.

**Dependencies.** Cluster storage. **Start today, in parallel with Action 1.**

---

## Action 6 — Author the 150–300 grounding phrases, freeze, and evaluate

**Action.** The one blocker in this repo that its own tooling explicitly cannot discharge. In order:
1. `uv run gdp ground sample-frames --dataset bdd100k --n 60 --seed 42` — writes `frames.json` with a
   `sampled_at` timestamp **before** you have seen any model prediction. This ordering is the bias
   control; do not invert it.
2. Hand-author 150–300 descriptive phrases against the generated overlays, spread across all four
   qualifier types (`spatial`, `attribute`, `relational`, **`negative` — mandatory**), into
   `data/grounding_eval/phrases.json`.
3. Fill in `data/grounding_eval/construction.md`'s two `_TODO_` fields — the authoring **date range**
   and the **drop count/rate** — **at authoring time, not backfilled afterward** (the file says so
   itself, and that is the whole point of the record).
4. `uv run gdp ground validate` → `uv run gdp ground freeze` (writes `phrases.lock.json`).
5. Submit `scripts/slurm/grounding_eval.slurm` to score zero-shot **and** the fine-tuned checkpoint.

**Why.** M3 is the "grounding" half of the project's name and one of the three keywords the Honda
contact named. It is also the project's most interesting *evaluation-design* story — a self-built set
with three documented bias controls, which is the H9 sanctioned exception.

**Current problem.** `data/grounding_eval/phrases.json` **does not exist**. The only `frames.json`
present samples 3 *fixture* image ids (dated 2026-09-02) — a rehearsal. `runs/04-grounding/` is empty.

**Expected outcome.** `runs/04-grounding/<ts>/grounding_comparison.json` with paired zero-shot and
fine-tuned grounding accuracy at IoU ≥ 0.5 over ≥150 phrases. Not guaranteed to improve — descriptive
phrases can regress after a class-name-driven fine-tune, and that regression would itself be worth
reporting.

**How to execute.** This is 6–10 hours of careful human work, not a command. Honour the ambiguity
rule in `construction.md` verbatim. Once frozen, `gdp ground evaluate` refuses to run against a
changed file (sha256 gate) — **so do not edit a phrase after seeing a prediction on it**; unfreeze,
edit, re-freeze before the next run, and say so.

**How to validate.** `gdp ground validate` passes with ≥150 phrases and coverage across all four
types; `phrases.lock.json` hash matches; `grounding_comparison.json` carries both numbers; every
artifact and every sentence about it is labelled **self-built**.

**Metrics affected.** **M3.**

**Resume impact.** "Designed and documented a 150+ phrase referring-expression evaluation set with
three explicit bias controls (frames sampled before authoring, hash-frozen phrases, mandatory
negatives); measured grounding accuracy at IoU ≥ 0.5, zero-shot vs fine-tuned." The evaluation-design
reasoning is the strongest interview material in the project.

**Dependencies.** Actions 1 and 3 (needs real frames and the fine-tuned checkpoint).

---

## Action 7 — Run the DriveLM chain: specs 06 → 07 → 08

**Action.** In strict order:
1. Spec 06's conversion (part of Action 5).
2. Submit `scripts/slurm/finetune_vlm.slurm` — LoRA fine-tune real `Qwen/Qwen2.5-VL-3B-Instruct`.
   Overfit gate (**fatal on failure**) → full LoRA run → generation check. **No evaluation here** — H2
   forbids scoring inside the training job.
3. Submit `scripts/slurm/vqa_eval.slurm` — `predict_base` → `predict_finetuned` → `score` →
   `failures`, against the real val split and the real adapter.

**Why.** Produces the **Stage-2 headline** (M4), the per-category breakdown that *is* this project's
risk-reasoning result (M5), and the hallucination diagnostic (M6). Without it the project is
detection-only and the "vision-language" half of its identity is unevidenced.

**Current problem.** No adapter exists; `runs/07-finetune-vlm/` is not even a directory. Spec 08's
only artifacts come from 8 hand-written fixture rows in `tests/fixtures/vqa_predictions/` — in that
comparison the "fine-tuned model" is a text file, and three of four categories are `null`.

**Expected outcome.** `runs/08-vqa/<ts>/comparison.json` with base and fine-tuned accuracy, plus
`metrics.json` and a `failures.md` scaffold.
**⚠️ Corrected 2026-09-24 ([SEQ-0143]/[SEQ-0144]):** "all four categories non-null" cannot be met
honestly. DriveLM's official tags (`extract_data.py`) put **planning** only under the paid
ChatGPT judge. Planning is therefore **reported as unscored** (user decision) and shown
qualitatively. Perception is scored via `[0]` + `[2]`, prediction via its yes/no `[0]` item,
behavior via `[0]`. The risk-reasoning claim must say exactly this.
**`final_score` will be `null` and that is correct and permanent** — `chatgpt` (weight 0.4) and
`match` (weight 0.2, fused with the same paid judge) require a live OpenAI call this project never
makes. Report `accuracy` + the language metrics, and state the omission every time.

**How to execute.**
1. Do **Action 8 first** so the official scorer's round-trip is verified before you depend on it.
2. Check `configs/train_vlm.yaml`: `lr: 1e-4`, `epochs: 2`, `batch_size: 1`, `grad_accum: 8`
   (real accumulation *is* implemented on the VLM path, unlike the detector path),
   `gradient_checkpointing: true`, `max_seq_len: 4096`, `overfit_loss_target: 0.05`.
3. Submit the fine-tune (24 h requested). **The overfit gate is fatal by design** — if it fails,
   fix the recipe before spending the allocation.
4. Submit the eval (12 h requested). Decoding is pinned greedy (`do_sample: false`, `num_beams: 1`,
   `max_new_tokens: 128`) so it is reproducible; do not change it between the base and fine-tuned
   passes or the comparison is invalid.
5. Fill in `failures.md`'s per-item commentary **by hand** — it is deliberately left empty because an
   LLM commenting on an LLM's failures is unfalsifiable.

**How to validate.** `comparison.json` has `num_items` in the thousands, all four per-category rows
non-null, `is_synthetic` false, the val-split sha256 recorded, and the H8 gate satisfied (base and
fine-tuned on the identical split with identical decoding).

**Metrics affected.** **M4, M5, M6.**

**Resume impact.** "LoRA fine-tuned Qwen2.5-VL-3B on DriveLM driving-scene QA; improved accuracy from
A to B on a scene-clean nuScenes-partitioned val split, with per-category results across perception /
prediction / planning / behavior and a documented hallucination diagnostic." Note the split caveat
must travel with the number.

**Dependencies.** Actions 5 and 8. Spec 07 must complete before spec 08 — no adapter, no comparison.

---

## Action 8 — Install `openai` and re-verify the official-scorer round-trip

> **✅ DONE 2026-09-23** — see [SEQ-0128]/[SEQ-0129]. All 31 tests in the four scorer files pass
> against the real vendored scorer; default suite 542 passed, 2 skipped. **The command below is
> wrong as written:** `uv sync` removes everything not in `uv.lock`, including the `demo` extra and
> the git-installed `language_evaluation`. Use
> `uv sync --extra drivelm-eval --extra demo && uv pip install "git+https://github.com/bckim92/language-evaluation.git"`.

**Action.** `uv sync --extra drivelm-eval`, then `uv run pytest tests/test_vqa_score.py
tests/test_vqa_compare.py tests/test_vqa_official.py tests/test_vqa_eval_cli.py -q`.

**Why.** 15 tests currently **skip** on this machine because `openai` is absent from `.venv` — it is
an import-only dependency of the vendored `evaluation_suit` (the repo never calls the API). Until it
is installed, the claim "round-trips through the actual vendored scorer, not a mock" is not
re-verifiable here, and Action 7's scoring step would be running an unexercised path against an
expensive cluster job's output.

**Current problem.** `uv run python -c "import openai"` → `ModuleNotFoundError`.
`language_evaluation` **is** installed and imports fine, so this is the only gap.

**Expected outcome.** All 15 previously-skipped tests execute and pass, restoring the default suite
to a genuinely complete run.

**How to execute.** One command on the laptop. If `language_evaluation` also turns out to be missing
on the *cluster*, install it there per `docs/drivelm-eval-scorer-deps.md`
(`pip install "git+https://github.com/bckim92/language-evaluation.git"` then
`language_evaluation.download('coco')`).

**How to validate.** `uv run pytest -q -rs` reports zero skips with the reason "vendored scorer deps
not installed".

**Metrics affected.** Validity of **M4, M5** (does not measure them).

**Resume impact.** None directly — it protects Action 7 from a late failure on a 12 h job.

**Dependencies.** None. **Five minutes, do it now.**

---

## Action 9 — Re-run the demo with real artifacts and record the video

**Action.** `uv run gdp demo -c configs/default.yaml -c configs/demo.yaml --checkpoint
runs/03-finetune/<ts>/checkpoint-N --adapter runs/07-finetune-vlm/<ts>/adapter`, confirm both honesty
badges flip away from "NOT the fine-tuned model (H8)", then record the video following
`docs/demo-recording.md`'s shot list.

**Why.** The charter's §13 deliverable list includes the demo video, and it is the only artifact a
recruiter can evaluate in thirty seconds. `specs/README.md`'s spec-09 note is explicit: **the video
that ships is recorded from the real-checkpoint run, not today's.**

**Current problem.** The demo runs today, but every panel is badged zero-shot / base-VLM. Recording
it now would produce a video of the *unfinished* system.

**Expected outcome.** A recording showing query→boxes from the fine-tuned detector, question→answer
from the LoRA-adapted VLM, and the attributed scene report — with each statement attributed to the
model that produced it (H6) and the report labelled a demo with no accuracy number (H7/H9).

**How to execute.** Laptop-native; the `--checkpoint`/`--adapter`/`--variant`/`--scenes` seams are
already built and need no code change. Copy the checkpoint and adapter down from Rudra first.

**How to validate.** Both badges show the real weights; the recording covers every item in
`docs/demo-recording.md`'s shot list.

**Metrics affected.** None — this is a demo and carries no number, by design.

**Resume impact.** A linkable artifact. Zero metric value; real screening value.

**Dependencies.** Actions 3 and 7.

---

## Action 10 — Regenerate the metrics tables, run `/claims-check`, rewrite the resume bullets

**Action.**
```bash
uv run gdp report snapshot && uv run gdp report render && uv run pytest -q
```
then commit `docs/metrics_snapshot.json` alongside the README, run `/claims-check` (or the
`claims-auditor` agent), flip the `specs/README.md` status rows to `done`, and update
`PROJECT_RESULTS_AUDIT.md` with the real numbers. Only then write resume bullets from the rendered
tables — never from memory or from prose.

**Why.** `specs/README.md`'s spec-10 note is explicit that when the runs land, **no code changes** —
the whole action is these three commands. And `CLAUDE.md` §9 makes `/claims-check` the gate on any
user-facing text.

**Current problem.** All four README tables read "Not yet measured", correctly. The audit
(`PROJECT_RESULTS_AUDIT.md` §7.5) lists nine claims that must not appear on a resume today.

**Expected outcome.** Four rendered tables carrying real numbers with their baselines, splits and
dates; a clean claims audit; resume bullets traceable to a file in `runs/`.

**How to execute.** Run the three commands; hand-editing a generated block fails
`tests/test_report_render.py`, which is the point. Then re-read §7.5's forbidden-claims table and
check each retired claim against its now-existing artifact.

**How to validate.** `uv run gdp report render --check` passes; `pytest -q` exits 0;
`/claims-check` returns clean; every resume number can be pointed to a `runs/<spec>/<ts>/*.json`
path within ten seconds.

**Metrics affected.** All — this is the publication gate, not a measurement.

**Resume impact.** Converts measurements into defensible claims. **Do not write a single bullet
before this action.**

**Dependencies.** Actions 2, 3, 4, 6, 7.

---

## What is deliberately NOT in this plan

Per the audit and the "limited time" constraint, the following were considered and **excluded**
because they do not change a metric or close an evidence gap:

- Refactoring, renaming, or restructuring anything. The code is lint-clean and well tested.
- Additional tests. 509 pass; the gap is measurements, not coverage.
- Spec 11 (temporal demo). It is a stretch spec that by design **carries no metric** (H9), and
  `specs/README.md`'s cut order names it first to cut.
- Rendering `docs/architecture.png`. GitHub renders the mermaid block natively; the omission is
  already documented.
- Any performance optimization of the model itself. Nothing suggests a bottleneck that changes a
  headline number, and Action 4's device fix is a *correctness* fix, not an optimization.
- Building a rule-based risk engine. Considered and rejected in the charter for good reason (H9);
  risk reasoning is measured through DriveLM's planning/behavior categories in Action 7.

---

## If time runs out

`specs/README.md`'s own cut order, which this plan endorses: **Actions 1 → 2 → 3 → 4 are the
irreducible core.** Completing only those yields "a fine-tuned, grounded, edge-deployed driving
detector with measured gains" — the charter's own checkpoint, and a genuinely defensible portfolio
project. Actions 5–7 (the VQA stage) are additive; Action 6 (grounding) sits between the two in
value. Never sacrifice 1–4 to start 7 early.
