# Reproduction

Two halves, and they are labelled differently on purpose.

| Half | Where it runs | Status |
|---|---|---|
| **Laptop path** — install, tests, smoke, both fixture data pipelines, README render check | Apple M4, 16 GB, MPS. No GPU, no dataset, no network beyond `uv sync` | **Executed from a clean clone; all steps exited 0.** `bash scripts/repro_clean_clone.sh` — the run and the defects its first attempt found are logged in `progress_report.md` |
| **Cluster path** — detector fine-tuning, VLM LoRA fine-tuning, full-split evaluation | IITB SLURM cluster | **Documented, not executed.** No run has happened yet; every wall-clock figure below is a `#SBATCH --time` *request*, not a measurement |

The distinction matters more than it looks. Claiming an unrun reproduction is a claim like any
other, and the SLURM time limits below are requests written when the scripts were authored — they
are upper bounds someone chose, not runtimes anyone observed. When these jobs run, replace them with
what `sacct` reports and say so.

---

## 1 · Laptop path (runnable today, offline)

```bash
bash scripts/repro_clean_clone.sh          # add --keep to inspect the temporary clone afterwards
```

Clones `HEAD` into a temp directory and runs, printing each exit code:

| Step | Command |
|---|---|
| install | `make install` (`uv sync`) |
| tests | `make test` (`pytest`, `model_heavy` deselected — see CLAUDE.md §4) |
| smoke | `make smoke` (import → config → device → fixture → CLI) |
| Stage-1 fixture data | `uv run gdp data prepare -c configs/default.yaml --dataset mini_bdd --split val` |
| Stage-2 fixture data | `uv run gdp data prepare-drivelm -c configs/default.yaml --dataset mini_drivelm --split both` |
| README tables | `uv run gdp report render --check` |

Everything here is offline and runs on synthetic fixtures. **No number produced by this path is a
result** — `tests/fixtures/` artifacts are stamped `is_synthetic: true` and the report generator
renders them as pending, not as numbers (H7).

To run the tests that load real Hugging Face weights (deselected by default because a 16 GB laptop
cannot afford them casually):

```bash
uv run pytest -m model_heavy
```

---

## 2 · Datasets (registration required, not downloadable by script)

| Dataset | Used by | Registration |
|---|---|---|
| BDD100K (images + labels) | specs 01–05 | `docs/bdd100k-download.md` |
| nuScenes `v1.0-trainval` + DriveLM answered train file | specs 06–08 | `docs/drivelm-download.md` |

Both need an account and a manual accept-terms step, so neither is scripted. Once they are on the
cluster's storage, point the configs at them:

```bash
uv run gdp data prepare        -c configs/default.yaml --dataset bdd100k --split val
uv run gdp data prepare-drivelm -c configs/default.yaml -c configs/drivelm.yaml \
                                --dataset drivelm --split both
```

**The DriveLM split is not the leaderboard split** — the challenge val/test answers are withheld
behind EvalAI, so this repo partitions DriveLM's *answered* train file by the official nuScenes
scene lists. Every artifact carries a `caveat` field saying so; see `specs/06-data-drivelm.md`.

---

## 3 · Cluster path (documented, not executed)

Order matters: each job consumes the previous one's output, and specs 03 and 08 cannot produce an
honest number without their baseline job having run first (H8).

| # | Job | Command | `--time` requested | Produces |
|---|---|---|---|---|
| 1 | Zero-shot baseline (spec 02) | `sbatch scripts/slurm/zeroshot_eval.slurm` | 04:00:00 | `runs/02-zeroshot/<ts>/metrics.json` — **the H8 baseline for Stage 1** |
| 2 | Detector fine-tuning (spec 03) | `sbatch scripts/slurm/finetune_detector.slurm` | 24:00:00 | `runs/03-finetune/<ts>/comparison.json` — the zero-shot → fine-tuned mAP delta |
| 3 | Grounding evaluation (spec 04) | `sbatch scripts/slurm/grounding_eval.slurm` | 04:00:00 | `runs/04-grounding/<ts>/grounding_comparison.json` |
| 4 | VLM LoRA fine-tuning (spec 07) | `sbatch scripts/slurm/finetune_vlm.slurm` | 24:00:00 | `runs/07-finetune-vlm/<ts>/adapter/` |
| 5 | VQA evaluation (spec 08) | `sbatch scripts/slurm/vqa_eval.slurm` | 12:00:00 | `runs/08-vqa/<ts>/comparison.json` + `metrics.json` + `failures.md` |

Every script is resumable and requests `--gres=gpu:1`; the partition name is marked `# EDIT` in each
file because it is cluster-specific. Deployment (spec 05) is **not** on this list — quantization,
ONNX export, and latency benchmarking run on the laptop, which is the edge target being measured.

Before submitting anything, read `specs/README.md`'s per-spec status notes: three of these five jobs
have a human prerequisite (dataset registration, and spec 04's 150–300 phrases must be authored by
hand before its job can run at all).

### Step 4 · fold the results into the README

When the jobs land, no code changes:

```bash
uv run gdp report snapshot     # runs/*/  -> docs/metrics_snapshot.json
uv run gdp report render       # snapshot -> README.md's generated blocks
uv run pytest -q               # the render gate re-checks the README
```

Then commit `docs/metrics_snapshot.json` together with the README, and re-run `/claims-check`
before any of those numbers reach a reader.

---

## 4 · What reproduction does *not* cover

- **The demo video** (`docs/demo-recording.md`) — a human recording step.
- **`docs/architecture.png`** — see `docs/architecture-render.md`; the README renders the mermaid
  source directly, so the PNG is only needed for slides.
- **The DriveLM official composite `final_score`** — two of its four weighted components need a
  paid, non-deterministic OpenAI judge, so it is reported as `null` with its composition and reason.
  Reproducing a number that depends on a paid judge's mood is not reproduction.
