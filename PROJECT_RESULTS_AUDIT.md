# Project Results Audit — Grounded Driving Perception

**Audit date:** 2026-09-03 · **Auditor:** Claude Code session (read-only verification pass)
**Repo state audited:** `main` @ `f756beb` ("Record the clean-clone reproduction result"), plus 5
untracked files (`PROJECT_WRITEUP.md`, `docs/laptop-demo.md`, 3 PARAM-Rudra guides).

> **Purpose of this file.** This is the single source of truth for *what has actually been measured*
> in this project, as distinct from what has been implemented, planned, or documented. Every number
> below was read out of a file in this repository or produced by a command run during this audit;
> the command or file path is named beside it. Nothing here is estimated, inferred, or carried over
> from prose.
>
> **The one-sentence verdict:** the engineering is complete and unusually well disciplined; **none of
> the four headline scientific results the project exists to produce has been measured**, because
> neither real dataset has been downloaded and no cluster job has ever been run.

---

## 1 · Original objective

### 1.1 What the project was for

From the charter (`grounded-driving-perception-spec.md` §1, immutable, unedited since 2026-07-10):

> Build a vision-language perception system for driving scenes that can (a) **find any road user or
> object described in natural language** … by adapting an open-vocabulary detector to driving data,
> and (b) **answer natural-language questions about the scene** … by fine-tuning a compact
> vision-language model on a public driving-QA benchmark; then (c) **prove deployability** by
> quantizing, exporting to ONNX, and publishing the accuracy-vs-latency curve.

### 1.2 Why it was built

Charter §3, corroborated by `PROJECT_WRITEUP.md` §1: this is a **portfolio/placement artifact**, not
a product. Its stated job is to fill the one empty territory in the author's nine-project portfolio —
object detection, grounding, and vision-language — and to map onto three specific Honda R&D job
descriptions (Job 6 DNN Development Engineer, Job 7 AI Research Engineer, Job 4 Applied Data
Scientist). The success condition is therefore **evidence that survives an interviewer's questions**,
not a shipped system.

### 1.3 The problem being solved (technical framing)

Closed-set detectors (YOLO/DETR fine-tunes) see only their fixed taxonomy; real roads always contain
out-of-list objects. Open-vocabulary detection localizes from language, and a VLM reasons about the
scene in language. The project's claim to non-triviality is that both are **domain-adapted with a
measured gain against a paired baseline**, rather than demonstrated zero-shot.

### 1.4 Expected outcome / definition of success

The charter is explicit and falsifiable (§12 "Metrics to report"; §11 build plan; `specs/README.md`
"The checkpoint rule"). Success = **four measured deltas**, each paired with its own baseline on the
same split:

1. Zero-shot → fine-tuned **detection mAP** on BDD100K val.
2. **Grounding accuracy** (IoU ≥ 0.5) on the self-built descriptive-phrase set, zero-shot vs fine-tuned.
3. Base → LoRA fine-tuned **DriveLM VQA accuracy** with per-category breakdown + failure analysis.
4. **Deployment**: p50/p95/p99 latency + FPS for fp32/INT8, with the **accuracy-vs-latency curve**.

Plus a partial-credit line the project drew itself (`specs/README.md`, "The checkpoint rule"): after
spec 05 the project is *already* "a fine-tuned, grounded, edge-deployed driving detector with
measured gains"; specs 06–08 are additive.

The project also imposed a constraint most portfolio projects do not: the **Honesty Register H1–H9**
(`CLAUDE.md` §2), which forbids any metric without its paired baseline (H8), any self-invented
benchmark (H9), and any fixture number presented as a result (H7).

### 1.5 What would make this genuinely impressive to a top-tier reviewer

Stated plainly, and *not* met today:

- A **real mAP delta** on BDD100K val (e.g. "zero-shot 21.4 → fine-tuned 34.8 mAP@[.5:.95], N=10k
  val images"), because that is the only number that proves the candidate can actually train and
  evaluate a detector rather than call one.
- A **real base-vs-LoRA VQA delta** on DriveLM with per-category rows, because that proves
  multimodal fine-tuning under a real official scorer.
- A **latency/FPS curve on a fair, like-for-like execution target** showing the quantization
  tradeoff, because that is the literal "edge deployment" line in Honda J6/J7.
- The engineering discipline (SDD loop, honesty register, generated-not-typed metrics tables,
  scene-clean splits, an answer-only label mask with assertion chains) is genuinely above the
  portfolio norm — but it is **supporting evidence, not a headline**. No recruiter screens on it;
  an interviewer rewards it only after a real number has earned the conversation.

---

## 2 · Is the project complete?

### 2.1 Verdict

> **Functionally complete but scientifically unvalidated.**
>
> Every planned component is implemented, tested, lint-clean, and end-to-end runnable on synthetic
> fixtures. **Zero of the four headline metrics has been measured on real data.** The project has
> produced a working *instrument* and has not yet taken a single *measurement*.

This is not a documentation gap — the repo says so itself, loudly and in the right places
(`README.md` Results section, `specs/README.md` per-spec status notes, `docs/reproduction.md`'s
"Documented, not executed" row). The audit confirms those self-assessments are accurate, not
modest.

### 2.2 Evidence for the verdict

| Check | Command / file | Result |
|---|---|---|
| Default test suite | `uv run pytest -q` | **exit 0**; 509 selected, ~15 skipped, 61 `model_heavy` deselected |
| Real-model inference path | `uv run pytest tests/test_detect.py -m model_heavy -q` | **2 passed, exit 0** — real `grounding-dino-tiny` loads and detects |
| Lint | `uv run ruff check .` / `ruff format --check .` | **All checks passed**; 127 files already formatted |
| README tables match snapshot | `uv run gdp report render --check` | "README generated blocks match the snapshot (4 blocks)" |
| Real datasets present | `du -sh data` / `find data -maxdepth 3` | **88 KB total** — only `data/grounding_eval/`. No BDD100K, no nuScenes, no DriveLM |
| Detector fine-tuning ever ran | `ls runs/03-finetune/*` | **5 timestamp directories, all empty** |
| Grounding eval ever ran | `ls runs/04-grounding/` | **empty** |
| VLM fine-tuning ever ran | `ls runs/` | **no `runs/07-finetune-vlm/` directory at all** |
| Stub/TODO scan in code | `grep -rn "TODO\|FIXME\|NotImplementedError\|placeholder"` on `src/ scripts/ configs/` | **no stubs**; the single `NotImplementedError` is a deliberate refusal-to-lie guard (`src/gdp/train/trainer.py:80`, rejects `grad_accum≠1` rather than silently ignoring it) |
| Code volume | `wc -l` | 9,941 LOC in `src/`, 7,677 LOC in `tests/` |
| Build narrative | `grep -c "^## \[SEQ-" progress_report.md` | **119 entries**, append-only |

### 2.3 Component-by-component status

| Spec | Component | Implemented | Tested | Ever executed on real data | Status |
|---|---|---|---|---|---|
| 00 | Scaffold: config→dataclass, seeding, device select, CLI, fixtures | ✅ | ✅ | n/a (laptop-native) | **Complete** |
| 01 | BDD100K → COCO converter, class↔prompt-span map, drop accounting | ✅ | ✅ | ❌ — only on the 4-image `mini_bdd` fixture (`runs/01-data/20260715-005027/stats.json`: 4 images, 10 boxes) | **Code complete, unexercised** |
| 02 | Zero-shot detect + COCO mAP + operating-point sweep | ✅ | ✅ | ❌ — `runs/02-zeroshot/20260726-214630/metrics.json` is `is_synthetic: true`, 4 images | **Code complete, no baseline measured** |
| 03 | Detector fine-tuning (overfit gate → full run → re-score → compare) + open-vocab forgetting probe | ✅ | ✅ | ❌ — **never run**; `runs/03-finetune/*` all empty | **Code complete, never executed** |
| 04 | Grounding phrase schema, validator, hash-freeze gate, frame sampler, IoU scoring, `ground evaluate/compare` | ✅ | ✅ | ❌ — **and the 150–300 phrases have never been authored**: `data/grounding_eval/phrases.json` does not exist; `frames.json` samples 3 *fixture* image ids | **Code complete, input artifact missing** |
| 05 | ONNX export, INT8 dynamic quantization, interleaved latency bench, accuracy-vs-latency curve | ✅ | ✅ | ❌ — ran, but on 4 synthetic 360×640 fixture images (`runs/05-deploy/20260725-163907/`) | **Executed on fixtures only** |
| 06 | DriveLM converter, vendored official nuScenes split, scene-leakage gate, stratified subsampler, Qwen chat formatting | ✅ | ✅ | ❌ — `runs/06-data/*` from `mini_drivelm` (4 synthetic scenes) | **Code complete, unexercised** |
| 07 | LoRA attach (anchored regex + frozen-vision-tower assertion), answer-only label mask, `VLMTrainer` w/ real grad accumulation | ✅ | ✅ (on a few-layer randomly-initialised Qwen2.5-VL) | ❌ — **never run** on real `Qwen/Qwen2.5-VL-3B-Instruct`; no adapter exists | **Code complete, never executed** |
| 08 | Vendored official DriveLM scorer (pinned+hashed), resumable prediction, per-category scoring, hallucination diagnostic, H8 compare gate, stratified failure sampler | ✅ | ⚠️ **15 scorer round-trip tests currently SKIP** — `openai` is not installed in `.venv` (import-only dependency of the vendored `evaluation_suit`) | ❌ — 8 hand-built fixture prediction rows | **Code complete; official-scorer path not currently re-runnable here** |
| 09 | Gradio demo: query→boxes, question→answer, attributed scene report, honesty badges | ✅ | ✅ | ⚠️ Runs today against fixtures with base/zero-shot weights; every panel badged "NOT the fine-tuned model (H8)". **Demo video not recorded.** | **Runnable; not the shippable version** |
| 10 | Metrics snapshot + README table generator, architecture diagram source, reproduction doc, honesty tests | ✅ | ✅ | ✅ (the generator itself works) — **and it correctly renders all four tables as "Not yet measured"** | **Complete and working as designed** |
| 11 | Temporal multi-frame demo (stretch) | ❌ draft only | — | — | **Not started (explicitly first to cut)** |

### 2.4 Placeholders, mocks, and honest gaps found

- **No hardcoded or faked results anywhere.** The report generator refuses to promote a fixture
  number: every artifact carries `is_synthetic: true`, and `gdp report snapshot` stamps those stages
  `synthetic` so `gdp report render` prints "Not yet measured" instead of a value. Verified by
  reading `docs/metrics_snapshot.json` against `README.md`'s generated blocks.
- **One genuine `_TODO_` placeholder**, correctly flagged as pending human work:
  `data/grounding_eval/construction.md` — the authoring date range and drop-rate fields.
- **`docs/architecture.png` does not exist** and is not claimed to (README embeds the `.mmd` as a
  mermaid block; `docs/architecture-render.md` explains the omission).
- **`final_score: null` in every spec-08 metrics file, unconditionally** — two of DriveLM's four
  official sub-metrics (`chatgpt`, weight 0.4; `match`, weight 0.2, internally fused with the same
  paid judge) require a live OpenAI API call this repo never makes. This is documented in
  `third_party/drivelm/PROVENANCE.md` and is the correct call, but it means **the official composite
  DriveLM score can never be reported by this project** — only `accuracy` and the language metrics.

### 2.5 Can the intended workflow run end to end?

**On fixtures: yes, verified.** `bash scripts/repro_clean_clone.sh` is recorded in
`progress_report.md` [SEQ-0115] as having run clean-clone → install → tests → smoke → both data
pipelines → README render check, all exit 0.

**On real data: unknown and untested.** Every cluster path is `#SBATCH`-scripted but has never been
submitted. `docs/reproduction.md` labels this correctly: *"Documented, not executed. No run has
happened yet; every wall-clock figure below is a `#SBATCH --time` request, not a measurement."*

---

## 3 · The metrics that determine success

Targets below come **only** from the charter (§12), the specs' acceptance criteria, or the published
literature the project itself cites. Where no target was ever defined, this file says so rather than
inventing one.

| # | Metric | What it measures | Why it matters here | Target defined in repo? | Resume-worthy range |
|---|---|---|---|---|---|
| M1 | **Zero-shot mAP@[.5:.95]**, BDD100K val, 10 classes | Pretrained Grounding-DINO-tiny out of the box | The H8 baseline; without it every Stage-1 claim is unfalsifiable | **No numeric target** — charter §11 W1 only says "record zero-shot mAP. This number is the bar" | n/a — a baseline is not impressive, it is *required* |
| M2 | **Fine-tuned mAP@[.5:.95]**, same split | Domain adaptation gain | **The Stage-1 headline.** The project's single most important number | **No numeric target defined anywhere in the repo** | A credible, defensible **+8 to +15 mAP** delta with the baseline stated. Any positive, honestly-measured delta on the full val split beats what exists today (nothing) |
| M3 | **Grounding accuracy** (IoU ≥ 0.5), self-built phrase set, zero-shot vs fine-tuned | Referring-expression localization on descriptive phrases | Differentiates this from a plain detector fine-tune; the "grounding" keyword Honda was named as wanting | **Set size only**: `grounding.min_phrases ≥ 150` (150–300), 4 qualifier types incl. mandatory negatives (`configs/default.yaml`, `specs/04`). No accuracy target | A **paired** zero-shot→fine-tuned delta on ≥150 phrases, presented as a self-built set with its documented drop rate. Talking point value > number value |
| M4 | **DriveLM VQA accuracy**, base vs LoRA, overall | Multimodal fine-tuning gain | **The Stage-2 headline** | **No numeric target defined** | Positive delta on the full val partition (thousands of QA pairs), with per-category rows |
| M5 | **DriveLM per-category accuracy** — perception / prediction / planning / behavior | Where the gain came from | **This is the project's entire "risk reasoning" claim** (`specs/README.md`, "Where risk assessment lives"). Planning/behavior rows *are* the risk benchmark | No target | All four categories non-null with meaningful N per category |
| M6 | **Ungrounded-tag rate** (hallucination diagnostic) | Object tags emitted that aren't in the scene | Honest failure reporting; the "does it hallucinate?" interview answer | No target — **explicitly self-defined**, labelled as a diagnostic not a metric (H9) | A *decrease* base→fine-tuned, always labelled self-defined |
| M7 | **Latency p50/p95/p99 + FPS** per variant (fp32-pt / fp32-onnx / int8-onnx) | Edge inference cost | The literal J6/J7 "edge device deployment" line | **Protocol targets only**: 50 warmup discarded, 200 timed, interleaved, p95/p99 reported, hardware recorded (`configs/default.yaml` `deploy:`) | Real-resolution (720×1280) numbers on a fair like-for-like target, with a **quantization speedup that is actually a speedup** |
| M8 | **Accuracy-vs-latency curve** | The tradeoff, not just the win | H8 given a shape: no latency claim without its accuracy point | Deliverable defined; no numeric target | A curve where INT8 trades a small, quantified mAP loss for a real latency win |
| M9 | **Peak memory (RSS)** per variant | Edge feasibility | Secondary deployment evidence | No target | Per-variant figures that actually differ (see §6.3 — today they cannot) |
| M10 | **Reproducibility** — clean clone → tests → smoke → pipelines | Engineering credibility | Supports every other claim | Binary: all steps exit 0 | ✅ **Already met** on the laptop half |

---

## 4 · Actual verified results

### 4.1 Headline metrics

| # | Metric | Actual verified result | Source | Conditions |
|---|---|---|---|---|
| M1 | Zero-shot mAP, BDD100K val | **Not measured / no verified result available** | — | Real BDD100K is not downloaded (`data/` = 88 KB) |
| M2 | Fine-tuned mAP + delta | **Not measured / no verified result available** | `runs/03-finetune/*` — 5 empty directories | Fine-tuning has never been executed |
| M3 | Grounding accuracy | **Not measured / no verified result available** | `runs/04-grounding/` empty; `data/grounding_eval/phrases.json` **does not exist** | The phrase set itself has never been authored |
| M4 | DriveLM VQA accuracy, base vs fine-tuned | **Not measured / no verified result available** | — | No real DriveLM data; no LoRA adapter exists |
| M5 | Per-category accuracy | **Not measured / no verified result available** | — | — |
| M6 | Ungrounded-tag rate | **Not measured / no verified result available** | — | — |
| M7 | Latency / FPS | **Measured, but on synthetic fixtures and with a methodology flaw — not a result.** See §4.2 | `runs/05-deploy/20260725-163907/latency.json` | See §4.2 |
| M8 | Accuracy-vs-latency curve | **Produced as a file** (`runs/05-deploy/20260725-163907/curve.png`) but plotted from fixture mAPs of 0.0 / 0.0 / 0.045 — carries no information | same run dir | 4 synthetic images |
| M9 | Peak RSS per variant | **Measured but not per-variant** — see §6.3 | same `latency.json` | — |
| M10 | Clean-clone reproduction (laptop half) | **✅ Achieved.** All steps exit 0 | `scripts/repro_clean_clone.sh`, logged `progress_report.md` [SEQ-0115]; independently re-confirmed this audit via `pytest`/`ruff`/`report render --check` | Apple M4, 16 GB, offline, fixtures only |

### 4.2 The one set of real measurements that exists (spec 05 latency)

**These are real timings of real models.** They are *not* a result, for two independent reasons
stated below. Source: `runs/05-deploy/20260725-163907/latency.json`, git sha `a4c2d4d`, created
2026-07-25T11:45:52Z.

| Variant | p50 (s) | p95 (s) | p99 (s) | FPS | Peak RSS (bytes) |
|---|---|---|---|---|---|
| fp32-pt | 1.658 | 2.636 | 3.128 | 0.557 | 5,026,185,216 |
| fp32-onnx | 3.244 | 4.832 | 5.400 | 0.284 | 5,026,185,216 |
| int8-onnx | 3.142 | 4.193 | 4.967 | 0.297 | 5,026,185,216 |

**Conditions:** onnxruntime 1.28.0, torch 2.13.0, `processor: arm`, 10 threads, batch size 1, image
360×640 (the *fixture* resolution — real BDD100K is 720×1280, i.e. **4× the pixels**), 50 warmup
iterations discarded, 200 timed iterations, variants interleaved. Paired accuracy from
`metrics.json` in the same directory: fp32-pt **mAP 0.0**, fp32-onnx **mAP 0.0**, int8-onnx **mAP
0.045** — on **4 synthetic images with 10 ground-truth boxes total**.

**Why this is not a result:**

1. **The accuracy half is meaningless.** 4 synthetic fixture images; the repo stamps it
   `is_synthetic: true` and refuses to render it (H7). A latency number whose paired accuracy is
   noise is a half-truth (H8).
2. **The three variants did not run on the same execution target.** Confirmed by reading the code,
   not inferred: `src/gdp/cli.py:1738` constructs the PyTorch detector with `device=cfg.device`,
   which is `"auto"` (`configs/default.yaml`) and resolves to **MPS** on this M4
   (`src/gdp/seed.py:37-42`), while `OnnxDetector` runs on onnxruntime's default **CPU** execution
   provider. So "fp32-pt 1.66 s vs int8-onnx 3.14 s" compares *GPU PyTorch against CPU ONNX* — it is
   not a quantization measurement at all. `collect_hardware_info` (`src/gdp/deploy/bench.py:126-141`)
   records versions, processor, thread count, batch size and image size but **not the execution
   device or provider**, so the artifact does not disclose the asymmetry either.

**This is a new finding of this audit.** It is not recorded in `specs/05-deploy-onnx-edge.md`,
`README.md`, or `progress_report.md` (verified by grep). It matters because it is exactly the
question a Honda deployment engineer would ask first, and the current artifact answers it wrongly.

### 4.3 The synthetic VQA numbers, for completeness

`docs/metrics_snapshot.json` → `stage2_vqa` carries `overall_accuracy: {before: 0.5, after: 1.0,
delta: 0.5}` and BLEU/ROUGE figures. **These are not results and must never be quoted.** They come
from `runs/08-vqa/20260726-204745/`, computed over **8 hand-written fixture rows** in
`tests/fixtures/vqa_predictions/base_mini.jsonl` and `finetuned_mini.jsonl` — the "fine-tuned model"
in that comparison is a text file, not a model. Three of four categories are `null`. The snapshot
correctly stamps the stage `synthetic` and the README renders it as pending. Recording it here only
so that nobody later mistakes it for a measurement.

### 4.4 Reproducibility status

| Path | Reproducible today? | Notes |
|---|---|---|
| Laptop: install, tests, lint, smoke, fixture pipelines, README render | **Yes — re-verified during this audit** | `pytest` exit 0, `ruff` clean, `report render --check` clean |
| Real Grounding-DINO inference | **Yes** — `pytest tests/test_detect.py -m model_heavy` → 2 passed | Downloads `grounding-dino-tiny` |
| Vendored DriveLM official scorer round-trip | **No, currently** — 15 tests skip: `openai` missing from `.venv` (import-only dependency). Fix: `uv sync --extra drivelm-eval` | `language_evaluation` **is** installed and imports fine |
| Full 61 `model_heavy` suite | **Not attempted in this audit** — deliberately, per `CLAUDE.md` §4's 16 GB memory rule | Only the `test_detect.py` subset was run |
| Any cluster job | **No** | Never submitted; no cluster artifacts exist |

---

## 5 · Expected vs actual

| Metric | Target (source) | Impressive range | **Actual verified** | Gap | Met? |
|---|---|---|---|---|---|
| M1 Zero-shot mAP | No number; "record it" (charter §11 W1) | n/a (baseline) | **Not measured** | 100% — no run | ❌ |
| M2 Fine-tuned mAP delta | No number defined | +8 to +15 mAP, paired | **Not measured** | 100% — never executed | ❌ |
| M3 Grounding accuracy | ≥150 phrases, 4 types, negatives mandatory (`configs/default.yaml`) | Paired delta on ≥150 phrases | **Not measured**; **0 phrases authored** | 100% — input artifact absent | ❌ |
| M4 VQA accuracy delta | No number defined | Positive delta, full val partition | **Not measured** | 100% — no data, no adapter | ❌ |
| M5 Per-category accuracy | 4 categories non-null | All 4, meaningful N | **Not measured** (fixture run: 3 of 4 null, N=2/category) | 100% | ❌ |
| M6 Hallucination rate | Self-defined diagnostic | Decrease, labelled | **Not measured** | 100% | ❌ |
| M7 Latency p50/p95/p99 + FPS | Protocol: 50 warmup, 200 timed, interleaved, tails reported | Fair like-for-like target, real resolution, INT8 actually faster | **Measured**: p50 1.658 / 3.244 / 3.142 s (fp32-pt / fp32-onnx / int8-onnx) — **but MPS vs CPU, at fixture resolution** | Protocol ✅; validity ❌ | ⚠️ Partial |
| M8 Accuracy-vs-latency curve | Deliverable exists | Real quantified tradeoff | **File exists**, plotted from mAP 0.0/0.0/0.045 | Carries no signal | ⚠️ Shell only |
| M9 Peak RSS per variant | No target | Differing per-variant figures | **Identical across all three** (process-global `ru_maxrss`) | Not per-variant | ❌ |
| M10 Reproducibility (laptop) | All steps exit 0 | — | **All steps exit 0** | None | ✅ |

**Score: 1 of 10 met. 1 partial. 8 not measured.**

---

## 6 · Why the results are where they are

### 6.1 Confirmed causes (direct evidence in the repo)

1. **Neither real dataset has ever been downloaded.** `data/` is 88 KB and contains only
   `grounding_eval/`. Both BDD100K and nuScenes+DriveLM require account registration and a manual
   terms acceptance, so neither is scriptable (`docs/bdd100k-download.md`, `docs/drivelm-download.md`).
   **This single fact blocks M1–M6 — six of the ten metrics — and therefore both headline deltas.**
2. **No cluster job has ever been submitted.** All five SLURM scripts exist
   (`scripts/slurm/*.slurm`) and none has run; `runs/03-finetune/*` holds five empty timestamp
   directories, and `runs/07-finetune-vlm/` does not exist at all. `docs/reproduction.md` states this
   directly. The wall-clock figures in that document are `#SBATCH --time` *requests*, not measurements.
3. **The hardware constraint is real and was correctly respected, not worked around.** An Apple M4
   with 16 GB cannot fine-tune Grounding-DINO or a 3B VLM (`CLAUDE.md` §4), and a `guard-bash` hook
   enforces it. `progress_report.md` [SEQ-0034]/[SEQ-0035] records a real incident where a bare
   `pytest` reached ~40 GB by reloading a real model seven times. The project's response — a
   `model_heavy` marker deselected by default — was the right engineering call and is *why* the
   laptop half is reliable; it is also why nothing beyond a fixture has been measured here.
4. **Spec 04's blocking input is a human task the tooling cannot discharge.** 150–300 descriptive
   phrases must be hand-authored against sampled real frames *before* the freeze-and-evaluate
   pipeline can run at all. `phrases.json` does not exist. `frames.json` exists but samples 3
   *fixture* image ids, dated 2026-09-02 — i.e. a fixture rehearsal, not the real sampling pass.
5. **Two of DriveLM's four official sub-metrics are structurally uncomputable in this project.**
   `chatgpt` (weight 0.4) and `match` (weight 0.2, fused with the same judge) require a live paid
   OpenAI call the repo never makes. `final_score` is therefore `null` unconditionally. This is
   documented and is the honest choice — but it means M4 can only ever be reported as *accuracy +
   language metrics*, never as the official composite, and that limitation must be stated whenever
   the number is used.
6. **The deployment benchmark compares different execution targets** (§4.2). Confirmed by code
   inspection: `cli.py:1738` + `seed.py:37-42` → MPS for PyTorch; onnxruntime default → CPU. This is
   why INT8 appears *slower* than fp32 PyTorch, an inversion that would otherwise look like a
   quantization failure.
7. **`peak_rss_bytes` is process-global, not per-variant.** `_peak_rss_bytes()`
   (`src/gdp/deploy/bench.py:41-44`) returns `resource.getrusage(RUSAGE_SELF).ru_maxrss` — a
   high-water mark for the whole process. Because all three variants are benchmarked in one
   interleaved process, all three report the identical 5,026,185,216 bytes. The field is therefore
   not a per-variant memory measurement and cannot be quoted as one.
8. **The official-scorer test path does not currently run on this machine.** 15 tests skip because
   `openai` is absent from `.venv`. Trivially fixed (`uv sync --extra drivelm-eval`), but until it
   is, the claim "round-trips through the real vendored scorer" is not re-verifiable here.

### 6.2 Likely causes (evidence-supported, not proven)

- **Sequencing, not capability, is the binding constraint.** 119 progress entries, 17 commits,
  ~10k LOC of source and 7.7k of tests were produced against fixtures while the two dataset
  registrations — the long-lead items — remained open. The critical path was dataset access from
  day one, and the build order put it last.
- **Absolute latency (1.7–3.2 s/image at 360×640) is far slower than `grounding-dino-tiny` would
  typically suggest.** Plausible contributors visible in the code: `detect_images` re-runs full
  image preprocessing and text tokenization inside every timed call, so these are end-to-end
  pipeline timings rather than model-forward timings; and the ONNX export is fixed-prompt with all
  10 class names in one pass. Neither has been isolated by measurement — **untested**.

### 6.3 Explicitly not tested

- Any behaviour on real BDD100K or real DriveLM/nuScenes data.
- Any GPU (CUDA) execution whatsoever.
- Batch sizes other than 1; any concurrency; any sustained-load or stress test.
- Real-resolution (720×1280) latency.
- The full 61-test `model_heavy` suite (only `test_detect.py`'s 2 were run this audit).
- Whether the fine-tuning loop converges on real data — the overfit gate has only ever been run
  against synthetic fixtures and a randomly-initialised few-layer Qwen.

---

## 7 · Resume readiness

### 7.1 Verdict

> **Not resume-ready as a results project.** It is currently defensible only as an
> *engineering-and-evaluation-design* project, and only with careful wording.
>
> The resume bullets this project was designed to earn — "improved detection mAP from X to Y by
> fine-tuning an open-vocabulary detector on BDD100K", "improved DriveLM VQA accuracy from A to B
> with LoRA" — **cannot be written today, because neither number exists.** Writing them anyway would
> be fabrication, and a competent interviewer would find out in one question ("what was your
> baseline mAP?").

### 7.2 What is already strong and defensible

- **Reproducibility.** A clean-clone script that installs, tests, smokes, runs both data pipelines
  and verifies the README's generated tables, all exit 0 — re-verified in this audit.
- **Evaluation-design discipline, which is genuinely unusual.** Metrics tables *generated* from
  hashed run artifacts and never typed by hand (hand-editing fails a test); a synthetic-vs-real
  stamp that mechanically refuses to promote a fixture number; scene-level leakage gates on the
  DriveLM split; a hash-freeze on the grounding phrase set so no phrase can be edited after seeing a
  prediction; an explicit register of nine honesty rules enforced by a reviewer agent.
- **Correctly identifying and documenting an uncomputable official metric** (`final_score: null`)
  rather than substituting a proxy. This is a strong interview story on its own.
- **Honest self-assessment.** The README's four tables all say "Not yet measured" with a named
  blocker. Very few portfolio repos do this. It is the reason this audit found no overclaims to
  correct.

### 7.3 Technically valid but not impressive

- 509 passing tests / ~10k LOC / 119-entry build narrative. Real work, real discipline — but
  **volume of tests is not a result**, and no recruiter screens on it.
- ONNX export + INT8 quantization *pipeline exists and runs*. The artifacts are real
  (`model.onnx`, `model.int8.onnx` on disk). But the accompanying numbers are fixture-derived and
  the comparison is target-asymmetric, so the pipeline is claimable and the **benchmark is not**.
- The Gradio demo runs — with base/zero-shot weights and honesty badges saying so.

### 7.4 Missing entirely

Every headline metric: M1–M6, plus a valid M7/M8/M9.

### 7.5 Claims that must NOT go on the resume today

| Do not claim | Why |
|---|---|
| Any mAP number, or "improved detection accuracy by N%" | M1/M2 not measured. No baseline exists |
| Any VQA accuracy number or delta, incl. the 0.5→1.0 in `metrics_snapshot.json` | 8 hand-written fixture rows; the "fine-tuned model" is a text file |
| Any grounding-accuracy number | The phrase set does not exist |
| "Fine-tuned Grounding-DINO" / "LoRA fine-tuned Qwen2.5-VL" | Neither fine-tuning run has ever been executed. The *code* exists; the *run* does not |
| "Achieved X FPS / N ms on edge hardware" | Fixture resolution, and MPS-vs-CPU asymmetry (§4.2) |
| "3× speedup from INT8 quantization" or any quantization win | The measurement shows INT8-ONNX **slower** than fp32-PyTorch — and even that comparison is invalid |
| "Reduced memory to N GB" | `peak_rss_bytes` is process-global and identical across variants |
| "Evaluated on the DriveLM benchmark" (unqualified) | Not the leaderboard split, and the official composite score is uncomputable here. Both facts must be stated |
| "Deployed" without the definition | H3: means quantized + exported + latency-benchmarked. Never "in a vehicle" |

### 7.6 What *could* honestly go on a resume today

Only process claims, and they are weak on their own:

> *Built a spec-driven, fully reproducible evaluation harness for open-vocabulary driving perception
> (Grounding-DINO + Qwen2.5-VL): ONNX/INT8 export pipeline, COCO-mAP and vendored official DriveLM
> scoring paths, scene-leakage-safe splits, and README metrics tables generated from hashed run
> artifacts rather than hand-entered. 500+ tests; clean-clone reproduction verified end to end.*

Truthful and verifiable. It also silently reveals that no model was actually trained — which is why
§8's action plan exists.

### 7.7 Evidence required before a strong claim can be made

| Claim wanted | Evidence required |
|---|---|
| Stage-1 mAP delta | Real BDD100K val + `runs/02-zeroshot/<ts>/metrics.json` (real) + `runs/03-finetune/<ts>/comparison.json` |
| Grounding accuracy | ≥150 authored + frozen phrases, `construction.md` `_TODO_`s filled at authoring time, + `runs/04-grounding/<ts>/grounding_comparison.json` |
| Stage-2 VQA delta | Real DriveLM conversion + a real LoRA adapter + `runs/08-vqa/<ts>/comparison.json` with all 4 categories non-null |
| Edge latency | A re-run at 720×1280 with **all variants on the same execution provider**, device recorded in `latency.json`, paired with real-split mAP |

---

## 8 · Bottom line

| Question | Answer |
|---|---|
| What did we build? | A complete, well-tested, honestly-instrumented evaluation harness for a two-stage grounded driving-perception system — with the honesty controls that make its future numbers trustworthy |
| Is it complete? | **Functionally complete, scientifically unvalidated.** All code done; no measurement taken |
| What did it achieve? | A reproducible laptop pipeline and a real ONNX/INT8 export path. **No headline metric** |
| What can we prove? | Reproducibility (M10) and that the code runs on synthetic fixtures and on a real `grounding-dino-tiny` forward pass |
| Strong enough for a top-tech resume? | **Not yet.** The engineering is above the portfolio norm; the results section is empty |
| Minimum remaining work? | Download two datasets; run five SLURM jobs; author 150+ phrases; fix the latency benchmark's device asymmetry. See `REMAINING_ACTION_PLAN.md` |

**No number in this audit was estimated. Anything not measured is written as "Not measured".**
