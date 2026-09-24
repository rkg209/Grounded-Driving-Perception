# PARAM Rudra — cluster guide for the Grounded Driving Perception project

**Scope of this file:** everything in this repo (`4_Honda`) that CANNOT run on the M4 laptop and
needs PARAM Rudra's A100 GPUs and larger memory/storage instead — per `CLAUDE.md` §4's hard
constraint. This is the project-specific companion to the two general files already in this repo
root, `PARAM_Rudra_Quickstart_24m1531.md` (generic login/SLURM basics for user `24m1531`, written
for a different project — "Pallet-CC") and `PARAM_Rudra_Training_Workflow.md` (that other project's
transfer/training checklist). Those two were written **before anyone had actually used this
cluster**, and §0.5 below corrects them where the real cluster disagreed. Where they conflict with
§0.5, **§0.5 wins** — it is the only part of this guide that came from commands that actually ran.

**Do not run anything in this file on the laptop.** `guard-bash` and CLAUDE.md §4 exist specifically
to stop that — training Grounding-DINO or Qwen2.5-VL-3B on a 16 GB M4 will hang it.

---

## 0 · What needs the cluster, and why

| Blocked step | Spec | Why it can't run on the laptop |
|---|---|---|
| Register + download real BDD100K | 01/02/03/04 | Multi-GB dataset; laptop has none of it |
| Zero-shot mAP on real BDD100K val | 02 | Full-split inference; the H8 baseline every later Stage-1 claim needs |
| Detector fine-tuning | 03 | CLAUDE.md §4: "cannot fine-tune Grounding-DINO... Ever" on this laptop |
| Grounding accuracy on the 150–300 real phrases | 04 | Needs spec 03's fine-tuned checkpoint + the authored phrases |
| Register + download real nuScenes v1.0-trainval + DriveLM | 06 | Multi-GB dataset; separate registration from BDD100K |
| VLM LoRA fine-tuning (real Qwen2.5-VL-3B) | 07 | CLAUDE.md §4: cannot fine-tune a 3B VLM on this laptop |
| Base-vs-fine-tuned VQA accuracy | 08 | Needs spec 06's real data + spec 07's real adapter |

Everything else (spec 05's ONNX export/quantize/latency benchmark, spec 09's demo, spec 10's
report generator) is laptop-native by design — the M4 *is* the edge target for 05 (H3). Don't move
those to the cluster; they'd stop being a benchmark of edge-class compute.

---

## 0.5 · Verified cluster facts — read before giving any server command

Every row below was observed on this cluster during the `3_LLM_from_scratch` sessions
(2026-09-17 → 2026-09-22) and each one cost real debugging time there. **This project has not
touched Rudra yet**, so nothing 4_Honda-specific below is verified — but the environment facts are.

### 0.5a How the work actually happens

- The user drives the cluster through **VS Code Remote-SSH** and is already logged in.
  **A Claude session cannot run commands on the server** — the laptop's SSH `ControlMaster` socket
  is not shared with the Remote-SSH connection. Claude gives a command block, the user pastes it
  into the Remote-SSH terminal, and pastes the output back.
- **One step at a time.** Small block → say what output to expect → wait for the paste.
- **No comment lines inside command blocks the user pastes.** Explanations go in prose.
  (`#!/bin/bash` and `#SBATCH` lines inside a job script are directives, not comments — say so.)
- For heredocs (`cat > f <<'EOF' … EOF`), tell the user to paste the whole block at once, and that
  a lingering `>` prompt means typing `EOF` + Enter.
- **Getting files in:** code via `git clone` / `git pull` (login nodes have internet). Gitignored
  files (datasets, checkpoints) are **dragged from Finder into the VS Code Explorer**. Verify with
  `sha256sum` on both ends.
- **Getting files out:** VS Code Explorer → right-click → Download (verified). `scp`/`rsync` from
  the Mac is **untested and may be blocked by the interactive captcha login** — do not assume it.

### 0.5b Facts

| Thing | Value |
|---|---|
| User / group | `24m1531` / `ai-at-ieor` |
| Home | `/home/IITB/ai-at-ieor/24m1531` — **group quota 1T/1T FULL, writes fail.** `lfs quota -hu` hides this; check `lfs quota -hg ai-at-ieor /home`. **Tighter than in Sept 2026:** a 74-byte `~/.kaggle/kaggle.json` failed with `write error: Disk quota exceeded` (2026-09-22), though `~/.bashrc` appends still worked. Assume home is unwritable and put config files under `$SCR/.config/` |
| Scratch (`$SCR`) | **`/scratch/IITB/ai-at-ieor/24m1531`**. `/scratch/24m1531` does **not** exist |
| Scratch quota | none; not backed up; files untouched 3 months are purged |
| Login-node internet | **yes** (GitHub, Hugging Face, astral.sh, PyPI, download.pytorch.org) |
| **GPU/compute-node internet** | **NO** — DNS fails. Downloads happen on the login node only |
| **`git` on compute nodes** | **not installed** — record the commit with `cat .git/refs/heads/main`. Consequence for this repo: `gdp.paths.git_sha()` shells out to `git rev-parse HEAD`, so **every artifact produced by `srun`/`sbatch` carries `"git_sha": "unknown"`** (confirmed in `runs/01-data/20260922-225014/stats_*.json`). Echo the commit into the job log and record it by hand |
| **C headers on compute nodes** | **missing** (`/usr/include/stdlib.h` absent). Fix: `cp -r /usr/include $SCR/sysroot/` once on a login node, then `export C_INCLUDE_PATH=$SCR/sysroot/include` in every job |
| GPUs | 2× **A100 80GB PCIe** per node, 30 nodes in partition `gpu` (max 4 days) |
| CPU partitions | `debug` (1 h), `small`, `medium`, `large`, `hm`, `serial` — `small` is verified working |
| Driver / CUDA | **550.54.14 / CUDA 12.4** → torch must be a **cu12x** build (cu126 verified). Never cu13x |
| System python | `/home/apps/miniconda3` 3.12.2 — **do not build venvs on it**; venvs on it die with `init_fs_encoding` on GPU nodes |
| Queue wait | `gpu` jobs started within seconds (2026-09-17) |

### 0.5c Rules, each learned by breaking it

1. **Never write to home.** Installers, caches, venvs, model weights, `TMPDIR` — all under `$SCR`.
2. **Use uv with a uv-managed Python on scratch**, not conda. `uv` is already installed at
   `$SCR/bin/uv`; `uv python install 3.12` is already done. Set `UV_NO_SYNC=1`.
3. **Compute nodes are offline** → pre-download every HF model on the login node into `$HF_HOME`,
   then run jobs with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
4. **Job scripts must export every env var explicitly** — they do not source `~/.bashrc`.
5. **`mkdir -p` the log directory before `sbatch`**, or SLURM silently fails to write logs.
6. Never run real work on a login node. `srun --pty` for short interactive checks, `sbatch` for jobs.
7. Always run a tiny test job before the real one.
8. Pull results off scratch promptly (3-month purge, no backup).

### 0.5c-bis Verified on 2026-09-22 (this project's first real use of Rudra)

- `uv sync` in this repo resolves **torch 2.13.0+cu130** — the wrong CUDA family for driver 550.
  The swap `uv pip install --reinstall torch torchvision --index-url
  https://download.pytorch.org/whl/cu126` gives **torch 2.14.0+cu126 / CUDA 12.6**, which matches
  what the other project ended up on. Redo it after every `uv sync`.
- `uv venv --python-preference only-managed --python 3.12` + `uv sync --extra drivelm-eval` took
  ~100 s on a warm uv cache. `openai` and `language_evaluation` come in with that extra.
- The `small` partition allocated a 30-minute CPU job (409040) in seconds.
- Kaggle config goes to `$SCR/.config/kaggle/kaggle.json` + `export KAGGLE_CONFIG_DIR` (home is
  unwritable). `uv pip install kaggle` works; a 6.38 GB dataset downloaded at ~16 MB/s in ~7 min
  on a login node.
- `nohup … &` on a login node survives fine for long downloads; a second `unzip` over an
  already-extracted tree exits 1 rather than clobbering, which is harmless.

### 0.5d The `~/.bashrc` block (already appended for `24m1531`; verify with `tail -20 ~/.bashrc`)

```bash
export SCR=/scratch/IITB/ai-at-ieor/24m1531
export UV_CACHE_DIR=$SCR/.cache/uv
export HF_HOME=$SCR/.cache/huggingface
export PATH=$HOME/.local/bin:$PATH
export PATH=$SCR/bin:$PATH
export XDG_CACHE_HOME=$SCR/.cache
export XDG_DATA_HOME=$SCR/.local/share
export XDG_CONFIG_HOME=$SCR/.config
export UV_PYTHON_INSTALL_DIR=$SCR/.local/share/uv/python
export PIP_CACHE_DIR=$SCR/.cache/pip
export TRITON_CACHE_DIR=$SCR/.cache/triton
export TMPDIR=$SCR/tmp
export UV_NO_SYNC=1
```

`echo $SCR` printing a scratch path is the precondition for every command in this file.

### 0.5e What this means for *this* repo's SLURM scripts — fix before first submit

All five scripts in `scripts/slurm/` were written on the laptop against assumptions this cluster
breaks. Each needs the same edits before it is submitted:

| Line as written | Problem | Fix |
|---|---|---|
| `export HF_HOME="${SCRATCH:-$HOME}/hf"` | `$SCRATCH` is not a variable on Rudra (`$SCR` is), so this resolves to **home, whose quota is full** — the job dies downloading weights | `export HF_HOME=$SCR/.cache/huggingface` with `SCR` exported in the script |
| (no offline vars) | compute nodes have no internet; HF will hang then fail | pre-download on the login node; add `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` |
| (no `C_INCLUDE_PATH`) | Triton compiles on the first CUDA forward pass and dies on the missing `stdlib.h` | `export C_INCLUDE_PATH=$SCR/sysroot/include` |
| `uv run …` | `uv` is only on `PATH` via `~/.bashrc`, which jobs don't source | `export PATH=$SCR/bin:$PATH` and the `UV_*` vars, explicitly |
| `#SBATCH --output=runs/02-zeroshot/slurm-%j.out` | relative path, and the directory may not exist yet | `mkdir -p` it before `sbatch` |
| `#SBATCH --partition=gpu` | placeholder, but happens to be correct | leave as `gpu`; confirm with `sinfo` |

---

## 1 · One-time environment setup on Rudra

Superseded by §0.5 — the conda recipe this section used to carry was never run and conflicts with
rule 0.5c-2. The verified path is uv + a uv-managed Python on scratch.

Clone the repo (login node — it has internet; `rsync` from the Mac is not the route here because
the code is in git):

```bash
cd $SCR && git clone <this repo's URL> 4_Honda && cd 4_Honda
```

`data/` and `runs/` are gitignored and stay that way — the datasets arrive per §2, on the cluster.

Create the venv and install (first run pulls CUDA wheels, ~12 min; later runs are cache-fast):

```bash
uv venv --python-preference only-managed --python 3.12
uv sync --extra drivelm-eval
```

If torch resolves to a `+cu13x` build, swap it (driver 550 = CUDA 12.4):

```bash
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Redo that swap after every `uv sync`.** Then the two extra dependencies that are not on PyPI in
the form this repo needs:

```bash
uv pip install "git+https://github.com/bckim92/language-evaluation.git"
uv run python -c "import language_evaluation; language_evaluation.download('coco')"
uv pip install nuscenes-devkit
```

Verify — base prefix must be under `$SCR`, CUDA must be 12.x:

```bash
uv run python -c "import sys, torch; print(sys.base_prefix, torch.__version__, torch.version.cuda)"
uv run python -c "import gdp; print(gdp.__version__)"
bash scripts/smoke.sh
```

Pre-download the model weights on the **login node** (compute nodes are offline):

```bash
uv run hf download IDEA-Research/grounding-dino-tiny
uv run hf download Qwen/Qwen2.5-VL-3B-Instruct
```

GPU check before trusting any job script:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --time=00:15:00 --pty bash
```

then, on the node:

```bash
C_INCLUDE_PATH=$SCR/sysroot/include HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2 · Register and download the datasets (human step, do this first)

Full instructions already live in this repo — this file just points at them, since they're the
actual authority and shouldn't be duplicated out of date:

- **BDD100K:** `docs/bdd100k-download.md` — registration portal, expected directory layout under
  `data/bdd100k/`, and the `det_20` label files spec 01's converter expects.
- **nuScenes + DriveLM:** `docs/drivelm-download.md` — nuScenes `v1.0-trainval` registration
  (separate from BDD100K's), DriveLM's answered train file, expected layout under `data/nuscenes/`
  and `data/drivelm/`, plus §4's note on the split this pipeline actually produces (not the
  DriveLM leaderboard split — read that before quoting any split-derived count).

Download straight onto the cluster, into `$SCR/4_Honda/data/…` — **never home** (§0.5b: home's
group quota is full, so a multi-GB write there fails outright). Both datasets are tens of GB and
the Mac has neither the headroom nor the need to hold them.

**Downloads run on a login node**, not in a job: compute nodes have no internet (§0.5b). A long
download on a login node is I/O, not compute, so it is within the login-node rules — but keep it to
one at a time and don't also run anything heavy there.

**BDD100K's own portal returns `403 Forbidden` (observed 2026-09-16).** The working substitute is
the Kaggle mirror **`awsaf49/bdd100k-dataset`** (7.4 GB, v1 dated 2020-09-11), chosen over
`solesensei/solesensei_bdd100k` after listing both through Kaggle's public API:

- awsaf49 has the flat official layout — `images/100k/{train,val,test}` = 70,000 / 10,000 / 20,000 —
  and **`labels/det_v2_{train,val}_release.json`**, the 2020 detection release our converter's
  class names (`pedestrian`, `motorcycle`, `bicycle`) come from.
- solesensei carries the **2018** labels (`person`, `motor`, `bike`) and re-splits train/test into
  `trainA/trainB/testA/testB` subfolders — it would need both a class remap and an un-shuffle.

Kaggle needs an API token (kaggle.com → Settings → API → Create New Token → `kaggle.json`), placed
at `~/.kaggle/kaggle.json` with `chmod 600` — one of the few things that must live in home, and
small enough that the quota tolerates it (the `~/.bashrc` appends in §0.5d succeeded the same way).
`uv pip install kaggle`, then `kaggle datasets download awsaf49/bdd100k-dataset -p $SCR/4_Honda/data/bdd100k_raw`.

**Verify before trusting it** — it is an unofficial mirror, and Kaggle's "CC0" label is wrong
(BDD100K's own academic/non-commercial license still applies). Check that the categories match
`gdp.config.BDD100K_CLASSES`:

```bash
jq -r '.[].labels[]?.category' labels/det_v2_val_release.json | sort | uniq -c
```

If the names differ from the 10 in `BDD100K_CLASSES`, **stop** — that is a converter change, which
needs a spec amendment and a `progress_report.md` entry, not a quiet `sed`. Record the mirror,
its version, and `sha256sum` of both label files in `docs/bdd100k-download.md` so the provenance of
every later mAP number is traceable (H8: "the same split" must be checkable, not asserted).

---

## 3 · Stage 1 — detector: zero-shot baseline → fine-tune → grounding eval

Run in this exact order; each later job's env var points at the previous job's output.

### 3a. Zero-shot baseline (spec 02) — the H8 pairing baseline

```bash
cd $SCR/4_Honda
mkdir -p runs/02-zeroshot
sbatch scripts/slurm/zeroshot_eval.slurm
squeue --me
```

**Before submitting**, apply §0.5e's edits to the script — without them the job writes weights into
the full home quota, has no internet to fetch them with, and dies in Triton on the missing C
headers. The `--partition=gpu` placeholder happens to be correct on Rudra; confirm with `sinfo`.
Submit a short test first (§0.5c rule 7): `SWEEP_LIMIT=20` with `--time=00:20:00` exercises the
whole chain in minutes.
Resumable: re-`sbatch` after a walltime/pre-emption kill and it skips already-completed steps (see
the script's own header comment).

Output: `runs/02-zeroshot/<ts>/metrics.json` on the real BDD100K val split — this is the number
every later Stage-1 claim sits beside (H8). Keep its path; you need it below.

### 3b. Detector fine-tuning (spec 03)

```bash
ZEROSHOT_METRICS=runs/02-zeroshot/<ts-from-3a>/metrics.json sbatch scripts/slurm/finetune_detector.slurm
```

Chain: overfit gate (fatal if it fails) → full fine-tune → re-score val at spec 02's threshold →
`compare` (the zero-shot → fine-tuned mAP delta) → the open-vocab forgetting probe (H9: demo only).
Output: `runs/03-finetune/<ts>/comparison.json` — the Stage-1 headline number.

### 3c. Grounding eval (spec 04) — needs a human step first

The 150–300 phrases must be **hand-authored** before this can run at all — not something either
machine can do for the other:

```bash
# on the laptop or cluster, doesn't need GPU:
uv run gdp ground sample-frames -c configs/default.yaml --dataset bdd100k --n <N>
# author phrases.json by hand against the sampled overlays, then:
uv run gdp ground validate  -c configs/default.yaml --phrases data/grounding_eval/phrases.json
uv run gdp ground freeze    -c configs/default.yaml --phrases data/grounding_eval/phrases.json
```

Once frozen and pushed to the cluster:

```bash
ZEROSHOT_METRICS=runs/02-zeroshot/<ts-from-3a>/metrics.json \
FINETUNED_CHECKPOINT=runs/03-finetune/<ts-from-3b>/checkpoint-N \
  sbatch scripts/slurm/grounding_eval.slurm
```

Output: `runs/04-grounding/<ts>/grounding_comparison.json`.

---

## 4 · Stage 2 — VLM: DriveLM data → LoRA fine-tune → VQA eval

### 4a. Real DriveLM conversion (spec 06) — CPU-only, no SLURM script needed

```bash
uv run gdp data prepare-drivelm -c configs/default.yaml -c configs/drivelm.yaml \
  --dataset drivelm --split both
```

Runs directly on a login node (CPU, no GPU needed) once §2's downloads exist — this is conversion,
not training, so it doesn't need a batch job. Output: `runs/06-data/<ts>/train.jsonl` +
`val.jsonl`. Keep the `train.jsonl` path; spec 07 needs it.

### 4b. VLM LoRA fine-tuning (spec 07)

```bash
TRAIN_JSONL=runs/06-data/<ts-from-4a>/train.jsonl sbatch scripts/slurm/finetune_vlm.slurm
```

Chain: overfit gate (fatal on failure) → full LoRA fine-tune → one generation sanity check.
**Deliberately no evaluation step here** (H2) — scoring happens only in spec 08, on its own
invocation. Output: `runs/07-finetune-vlm/<ts>/adapter/`.

### 4c. VQA evaluation + failure analysis (spec 08)

```bash
VAL_JSONL=runs/06-data/<ts-from-4a>/val.jsonl \
ADAPTER=runs/07-finetune-vlm/<ts-from-4b>/adapter \
  sbatch scripts/slurm/vqa_eval.slurm
```

(Check the script's own header for its exact required env var names before submitting — it follows
the same `: "${VAR:?...}"` pattern as 3b/4b.) Chain: `predict_base` → `predict_finetuned` → `score`
→ `failures`. Output: `runs/08-vqa/<ts>/comparison.json` — the Stage-2 headline number, including
the `planning`/`behavior` per-category rows that **are** the risk-reasoning result (no separate
risk engine — `specs/README.md`'s "Where risk assessment lives" section). Note `final_score` is
`null` unconditionally in every run — two of DriveLM's four sub-metrics need a live paid OpenAI
call this repo never makes (documented in `third_party/drivelm/PROVENANCE.md`).

---

## 5 · Pulling results back to the laptop

Only the small stuff — `metrics.json`/`comparison.json`/logs, never full checkpoints unless you
specifically need them locally.

**Use VS Code Explorer → right-click the folder → Download.** That route is verified (it goes over
the existing Remote-SSH connection, so the captcha never comes up). `scp`/`rsync` from the Mac is
**untested against Rudra's interactive captcha login** — don't build a step around it.

For a single small JSON, pasting `cat`'s output back into the chat also works and leaves a
`sha256sum` in the log as proof it arrived intact — but that is a supplement to Download, not a
replacement.

Then, on the laptop:

```bash
uv run gdp report snapshot -c configs/default.yaml
uv run gdp report render
uv run pytest -q
```

commits `docs/metrics_snapshot.json` + the regenerated README tables — per `specs/README.md`'s spec
10 status notes, this is the *only* step needed once real numbers land; no code changes.

If you do need a real checkpoint locally (e.g. to re-run the demo with `--checkpoint`/`--adapter`
against real artifacts instead of the zero-shot/base badges):

use the same VS Code Download on `runs/03-finetune/<ts>/checkpoint-N`, then `sha256sum` it on both
ends before trusting it (the 3_LLM_from_scratch sessions verified this route for a LoRA adapter;
a detector checkpoint is bigger but the mechanism is the same).

---

## 6 · Housekeeping specific to this project

- **Never commit `data/`, `runs/`, checkpoints, or `*.onnx`** pulled back from the cluster — the
  laptop's `.gitignore`/`guard-bash` hook already blocks this; don't work around it.
- `HF_HOME` is set to `${SCRATCH:-$HOME}/hf` inside every SLURM script here, which is **wrong on
  Rudra** — `$SCRATCH` doesn't exist there, so it resolves to the full home quota. §0.5e has the fix;
  apply it before the first submit. Weights are Grounding-DINO ~700 MB and Qwen2.5-VL-3B ~7 GB.
- Scratch's 3-month purge policy applies to checkpoints too — pull anything you'd be upset to lose
  back to the laptop promptly after a run finishes. **Not** to home: it is full (§0.5b), which is
  where `PARAM_Rudra_Training_Workflow.md` §1's advice breaks down.
- Every SLURM script here is resumable by re-`sbatch`ing after a pre-emption/walltime kill — that's
  normal on a shared cluster, not a failure to debug.
- Before any number from `runs/` reaches a README, a write-up, or gets spoken out loud: run
  `/claims-check` (or the `claims-auditor` agent) per CLAUDE.md's Honesty Register.
