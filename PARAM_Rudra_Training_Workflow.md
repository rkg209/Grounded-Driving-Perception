# PARAM Rudra — Pallet-CC Training Workflow

Concrete, project-specific version of `PARAM_Rudra_Quickstart_24m1531.md`.
Scope: **training only** — submit a batch job on Rudra's A100 GPUs, then pull
the trained checkpoint back to this Mac for testing/eval (which stays local).

**User:** `24m1531` · **Login:** `ssh 24m1531@paramrudra.iitb.ac.in -p 4422`
**Code lives at:** `/scratch/24m1531/Pallet-CC/` (scratch = fast I/O, jobs run here;
not backed up, 3-month purge on untouched files — pull checkpoints out promptly)

---

## 0. One-time: log in and set password

```bash
ssh 24m1531@paramrudra.iitb.ac.in -p 4422
```
- Solve the ASCII captcha, then enter the temp password from `rudrasupport@iitb.ac.in`.
- You'll be forced to set a new password on first login. Expires every 90 days.

---

## 1. One-time: create the conda env on Rudra

Do this once, on the **login node** (never run training itself on the login node).

```bash
module load miniconda
conda create --name pallet_env python=3.10 -y
conda activate pallet_env
```

You'll transfer `requirements.txt` in step 2. Once it's on the cluster, install
everything **except the eval/render-only packages** (`vtk`, `py3dbp`,
`anthropic` — not needed for training, and `vtk` is slow/fragile to build
headless on a cluster node):

```bash
cd /scratch/24m1531/Pallet-CC
grep -vE '^(vtk|py3dbp|anthropic)$' requirements.txt > requirements-train.txt
pip install -r requirements-train.txt --no-cache-dir
```

Note: `tianshou` is **not** in `requirements.txt` on purpose — it's vendored
in `./tianshou/` and travels with the code copy in step 2. Do not
`pip install tianshou`.

Verify GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. Transfer code to Rudra (run from your Mac terminal)

Only what training actually needs — skip the local venv, logs/output
(regenerated), and the eval-only / thesis-artifact folders (`eval/`, `data/`,
`results/`, `docs/`, `report/`) since we're training-only for now:

```bash
rsync -avz -e "ssh -p 4422" \
  --exclude 'gopt_env' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude '.DS_Store' \
  --exclude '.ruff_cache' \
  --exclude '.claude' \
  --exclude '.env' \
  --exclude 'logs' \
  --exclude 'output' \
  --exclude 'eval' \
  --exclude 'data' \
  --exclude 'results' \
  --exclude 'docs' \
  --exclude 'report' \
  ~/MTP/Pallet-CC/ 24m1531@paramrudra.iitb.ac.in:/scratch/24m1531/Pallet-CC/
```

This carries over: `arguments.py`, `tools.py`, `model.py`, `render.py`,
`mycollector.py`, `masked_ppo.py`, `masked_a2c.py`, `ts_train.py`,
`ts_test.py`, `envs/`, `tianshou/`, `cfg/`, `requirements.txt`.

Re-run the same command any time you change code locally — `rsync` only
pushes the diffs.

---

## 3. Write the SLURM batch script (once, on Rudra)

`cfg/config.yaml` uses `num_processes: 8` parallel envs, so request enough
CPU cores to back them plus the main process. Create
`/scratch/24m1531/Pallet-CC/train_job.sh`:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=pallet_train
#SBATCH --output=/scratch/24m1531/Pallet-CC/logs/slurm.%J.out
#SBATCH --error=/scratch/24m1531/Pallet-CC/logs/slurm.%J.err

module purge
module load miniconda
conda activate pallet_env

cd /scratch/24m1531/Pallet-CC
mkdir -p logs

python ts_train.py --config cfg/config.yaml
```

Swap `cfg/config.yaml` for `cfg/config_v5.yaml` for the random-box-size V5
run. Adjust `--time` (max `4-00:00:00`) and `--gres=gpu:2` if you want both
A100s on the node (only useful if `ts_train.py`/tianshou is set up for
multi-GPU — otherwise stick to `gpu:1`).

You can create this file directly on the cluster with an editor (`nano`,
`vim`) after logging in, or write it locally and `rsync` it over with the
command in step 2 (it'll be picked up automatically next time you sync, or
copy it individually):
```bash
scp -P 4422 train_job.sh 24m1531@paramrudra.iitb.ac.in:/scratch/24m1531/Pallet-CC/
```

---

## 4. Submit and monitor the job

```bash
ssh 24m1531@paramrudra.iitb.ac.in -p 4422
cd /scratch/24m1531/Pallet-CC
sbatch train_job.sh
squeue --me                                   # PD = pending, R = running
tail -f logs/slurm.<jobid>.out                # live stdout
sacct -j <jobid>                              # resource usage once it finishes
scancel <jobid>                               # kill if needed
```

`ts_train.py` writes each run's checkpoints/TensorBoard logs to
`logs/<run_name>/` (run name encodes pallet size, scheme, `k_placement`,
`k_max`, `box_type`, algo, seed, optimizer, timestamp) — that directory is
what you'll pull back in step 5. `policy_step_final.pth` and
`policy_step_best.pth` are the checkpoints you care about.

---

## 5. Pull the trained checkpoint back to your Mac

Find the run folder name first (`ls logs/` on Rudra), then from your **Mac terminal**:

```bash
scp -r -P 4422 24m1531@paramrudra.iitb.ac.in:/scratch/24m1531/Pallet-CC/logs/<run_name> ~/MTP/Pallet-CC/logs/
```

or, if the job is still running and you want to sync incrementally:
```bash
rsync -avz -e "ssh -p 4422" \
  24m1531@paramrudra.iitb.ac.in:/scratch/24m1531/Pallet-CC/logs/<run_name>/ \
  ~/MTP/Pallet-CC/logs/<run_name>/
```

Scratch is **not backed up and purges files untouched for 3 months** — don't
leave the only copy of a checkpoint sitting there.

---

## 6. Test locally on the Mac

No cluster-specific code needed — the checkpoint is portable. Use the
existing local `gopt_env` (already has CPU/MPS PyTorch):

```bash
./gopt_env/bin/python ts_test.py --config cfg/config.yaml \
  --ckp logs/<run_name>/policy_step_final.pth

# or single-episode visualization:
# edit CONFIG_FILE/CHECKPOINT constants at the top of eval/test_single.py first
./gopt_env/bin/python eval/test_single.py
```

---

## Cheat sheet

| Task | Command |
|---|---|
| Log in | `ssh 24m1531@paramrudra.iitb.ac.in -p 4422` |
| Sync code → Rudra | `rsync -avz -e "ssh -p 4422" --exclude ... ~/MTP/Pallet-CC/ 24m1531@paramrudra.iitb.ac.in:/scratch/24m1531/Pallet-CC/` |
| Submit training job | `sbatch train_job.sh` |
| Check jobs | `squeue --me` |
| Live log | `tail -f logs/slurm.<jobid>.out` |
| Cancel job | `scancel <jobid>` |
| Pull checkpoint → Mac | `scp -r -P 4422 24m1531@...:/scratch/24m1531/Pallet-CC/logs/<run_name> ~/MTP/Pallet-CC/logs/` |
| Test locally | `./gopt_env/bin/python ts_test.py --config cfg/config.yaml --ckp logs/<run_name>/policy_step_final.pth` |
