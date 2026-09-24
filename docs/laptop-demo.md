# Laptop demo — what you can run today, offline

This is the "show me it works" guide for an **Apple M4, 16 GB, MPS** laptop. Everything here runs
without the cluster, without BDD100K, without nuScenes, and without a network connection once the
two Hugging Face models are cached.

**Read this first, and say it out loud when demoing:** every scene below is a **synthetic fixture**
(`tests/fixtures/mini_bdd`, 4 generated images), and every checkpoint is the **zero-shot detector**
and the **base VLM** — no fine-tuned artifact exists yet, because specs 03 and 07 are cluster-blocked.
So **nothing here is a result** (H7). The mAP this pipeline prints on fixtures is 0.0000 and that is
the *correct* output for coloured rectangles — it demonstrates plumbing, not accuracy. The demo UI
badges all of this on screen; don't undercut it in narration.

---

## 0 · Prerequisites (once)

```bash
cd /Users/rahul/Placement/Project/4_Honda
uv sync
bash scripts/smoke.sh          # expect: SMOKE OK
```

The demo lazily downloads two models on first use (~700 MB + ~7 GB). Both are already in
`~/.cache/huggingface/hub` on this machine:

- `IDEA-Research/grounding-dino-tiny` — Stage 1
- `Qwen/Qwen2.5-VL-3B-Instruct` — Stage 2

---

## 1 · The 2-minute demo — the integrated Gradio app (spec 09)

The one to show a person. Query → boxes, question → answer, and an **attributed** scene report.

```bash
uv run gdp demo -c configs/default.yaml -c configs/demo.yaml
```

Then open **http://127.0.0.1:7860**. Startup prints:

```
scenes: .../tests/fixtures/mini_bdd
detector: IDEA-Research/grounding-dino-tiny (pt)
VLM backend: Stage 2: local, bf16 on mps
```

What to click, in order:

1. **Pick a scene** (`scene_000`…`scene_003`) or use the image-upload box to run on your own dashcam
   frame — more convincing in front of someone than the four synthetic fixtures.
2. **Detect.** Type a free-text query (e.g. `pedestrian crossing the road. parked car.`) — this is
   the open-vocabulary part; it is not a fixed class list. Drag the **threshold** slider and watch
   boxes re-filter without re-running the model.
3. **Ask a question** (e.g. *"Is it safe for the ego vehicle to proceed?"*). The first question
   lazy-loads Qwen2.5-VL-3B from disk — **expect a visible one-off pause**; later questions skip it.
4. **Scene report.** Each line is tagged with the model that produced it — detector vs. VLM. That
   attribution is the H6 point of the whole panel: **two separate models, never one joint system.**

Useful flags: `--port 7873`, `--scenes <dir>`, `--share`, and `--checkpoint` / `--adapter` (which
flip the honesty badges once real fine-tuned artifacts exist).

Stop it with `Ctrl-C`.

---

## 2 · The CLI pipeline — Stage 1, end to end (specs 02/05)

Shows the actual engineering rather than the UI. Each command writes a timestamped
`runs/<spec>/<ts>/` directory with full provenance (config, checkpoint, split, image count, git SHA,
threshold and how it was chosen).

```bash
# detect -> predictions.json   (dominated by the one-off model load)
uv run gdp detect   -c configs/default.yaml --dataset mini_bdd --split val

# score  -> metrics.json       (no model load; returns immediately)
uv run gdp evaluate -c configs/default.yaml --dataset mini_bdd --split val
```

Expected output — and note the zeros are the honest answer on synthetic shapes:

```
mini_bdd/val: 4 images, 4 detections (box_threshold=0.25) -> runs/02-zeroshot/<ts>/predictions.json
mini_bdd/val: mAP=0.0000 mAP50=0.0000 P=0.000 R=0.000 (threshold=0.25) -> runs/02-zeroshot/<ts>/metrics.json
```

The written `metrics.json` carries `is_synthetic: true`, which is why the README's tables render this
as *pending*, never as a number.

### Edge deployment — the part the laptop is actually the right machine for (spec 05, H3)

The M4 *is* a legitimate edge target, so this stage runs here by design rather than as a fallback —
export and quantization are the real artifacts, and the latency benchmark below measures the real
hardware. (Both still run against fixture images, so the *accuracy* half of the accuracy-vs-latency
curve remains pending until BDD100K val exists — H8.)

```bash
uv run gdp deploy export   -c configs/default.yaml -c configs/deploy.yaml --dataset mini_bdd --split val
uv run gdp deploy quantize -c configs/default.yaml -c configs/deploy.yaml
```

`export` takes a couple of minutes and prints `fixed_prompt=True` — the ONNX graph is frozen to the 10-class
prompt and is **no longer open-vocabulary**; that limitation is written into the
`export_result.json` sidecar so it travels with the artifact. `quantize` shows the real size drop:

```
quantized runs/05-deploy/<ts>/model.int8.onnx (693713474 -> 181866276 bytes)
```

Optional and **slow (10 min+, don't run it live in front of anyone)** — the interleaved p50/p95/p99
latency and peak-RSS benchmark across all three variants, plus the accuracy-vs-latency curve:

```bash
uv run gdp deploy bench             -c configs/default.yaml -c configs/deploy.yaml --dataset mini_bdd --split val
uv run gdp deploy evaluate-variants -c configs/default.yaml -c configs/deploy.yaml --dataset mini_bdd --split val
```

"Edge-deployed" here means **quantized + ONNX-exported + latency-benchmarked on edge-class compute**.
It has never meant deployed in a vehicle (H3).

---

## 3 · Proving the repo is healthy (specs 00/10)

```bash
uv run pytest -q                      # 509 collected, 507 passed, 2 skipped (SEQ-0116)
uv run ruff check .                   # All checks passed!
uv run gdp report render --check      # README tables match docs/metrics_snapshot.json
bash scripts/repro_clean_clone.sh     # clones HEAD to a temp dir and repeats install/test/smoke
```

`uv run pytest -q` deliberately **never loads real model weights** — those tests are gated behind
`uv run pytest -m model_heavy` (CLAUDE.md §4; a bare run once took the 16 GB laptop to ~40 GB).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Port 7860 in use | `uv run gdp demo ... --port 7873` |
| Demo process left running | `pkill -f "gdp demo"` |
| First question in the demo hangs ~60 s | Expected — Qwen2.5-VL-3B is lazy-loaded on first use |
| `HF_TOKEN` warning on `gdp detect` | Harmless rate-limit notice; weights are already cached |
| `gdp deploy quantize` can't find a model | Run `gdp deploy export` first — quantize reads the most recent export |

---

## What you cannot demo on this laptop

Not a limitation to apologise for — it is CLAUDE.md §4, stated up front:

- **Detector fine-tuning (spec 03)** and **VLM LoRA fine-tuning (spec 07)** — cluster only.
- **Any real metric**, because BDD100K and nuScenes/DriveLM are not downloaded yet. See
  `docs/reproduction.md` §2–3 and `specs/README.md`'s per-spec status notes for exactly what is
  blocking each number.
- **Grounding accuracy (spec 04)** — additionally blocked on 150–300 hand-authored phrases, a human
  curation step this repo's tooling deliberately cannot discharge for itself.
