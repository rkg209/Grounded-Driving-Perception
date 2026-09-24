# PARAM Rudra — My Working Guide
**Username:** `24m1531`
**Cluster:** paramrudra.iitb.ac.in (port **4422**)
**Home:** `/home/24m1531/` (50 GB quota) | **Scratch:** `/scratch/24m1531/` (200 GB quota, files unused >3 months are auto-deleted)

This is a personal condensed workflow guide — login → transfer code/data → run DL/RL training on GPU → pull results back to your Mac.

---

## 1. Logging In

### From Mac (built-in Terminal, no install needed)
```bash
ssh 24m1531@paramrudra.iitb.ac.in -p 4422
```
- You'll be shown a small ASCII **captcha challenge** first — type the string exactly as shown, then enter your password.
- **First login:** you'll be forced to set a new password (temporary password comes via email from `rudrasupport@iitb.ac.in`).
- Passwords expire every **90 days** — you'll be prompted to change it on login when it does.
- Forgot password → raise a ticket at `https://paramrudra.iitb.ac.in/support` (do **not** try to reset any other way).

### GUI option (if you ever want a desktop-like SSH client on Mac)
- Not needed on Mac — Terminal's `ssh` is sufficient. (WinSCP/PuTTY/MobaXterm are Windows-only tools mentioned in the manual; macOS's native `ssh`/`scp`/`sftp` covers everything.)

### A few login-node rules
- You land on a **login node** — this is only for editing, compiling, submitting jobs, and transferring files.
- **Never run your training job directly on the login node** — it will get killed. Always submit via SLURM or request an interactive/compute session (see §4).
- Login nodes are shared across all users — be light on CPU/RAM there.

---

## 2. Moving Code & Data (Mac ↔ Rudra)

Use `scp` from your **Mac terminal** (not from inside the cluster), always with `-P 4422`.

**Upload code/dataset to Rudra:**
```bash
scp -r -P 4422 ~/my_project 24m1531@paramrudra.iitb.ac.in:/home/24m1531/
```

**Download results/checkpoints back to Mac:**
```bash
scp -r -P 4422 24m1531@paramrudra.iitb.ac.in:/scratch/24m1531/my_project/outputs ~/Downloads/
```

**For frequent syncing, `rsync` works too (native on Mac) and is faster for repeated transfers:**
```bash
rsync -avz -e "ssh -p 4422" ~/my_project/ 24m1531@paramrudra.iitb.ac.in:/home/24m1531/my_project/
```

### Where to put things
| Data type | Location | Why |
|---|---|---|
| Code, scripts, small configs | `/home/24m1531/` | Backed-up-by-you, persistent, small quota |
| Training data, checkpoints, logs (job I/O) | `/scratch/24m1531/` | Larger quota, faster I/O, but **not backed up** and purged after 3 months of inactivity |

**Rule of thumb:** run jobs out of `/scratch/24m1531/`, copy final model weights/results you care about back to `/home/24m1531/` (or straight to your Mac) once training finishes.

---

## 3. First-Time Environment Setup (Conda for DL/RL)

```bash
# On a login node
module load miniconda
conda create --name rl_env python=3.10 -y
conda activate rl_env
```

Rudra also ships **pre-built conda environments** you can just `module load` instead of building your own (all include GPU support):

| Framework | Module name |
|---|---|
| PyTorch (GPU) | `Pytorch-gpu` (2.2.1) |
| TensorFlow (GPU) | `Tensorflow-gpu` (2.15.0) |
| Keras (GPU) | `Keras-gpu` (3.0.5) |
| Distributed (Horovod) | for TF/PyTorch multi-node training |
| RAPIDS (data science) | `Rapids` (21.06) |

```bash
module avail          # see everything available
module load Pytorch-gpu
python -c "import torch; print(torch.cuda.is_available())"
```

If you need your own custom stack (e.g., specific RL libraries like `stable-baselines3`, `gymnasium`, `wandb`):
```bash
module load miniconda
conda activate rl_env
pip install torch stable-baselines3 gymnasium wandb --no-cache-dir
```
Do this install step **once**, on the login node — not inside every job script (it just needs to persist in your conda env).

---

## 4. Getting GPU Access

Rudra's GPU partition has **30 nodes**, each with **2× NVIDIA A100 (80GB HBM2e)**, 48 cores, 192GB RAM.

### Option A — Quick interactive GPU session (for debugging/testing your training script before a full run)
```bash
salloc --nodes=1 --time=1:00:00 --gres=gpu:1 --partition=gpu
squeue --me                # find which node you were assigned, e.g. gpu007
ssh gpu007                 # hop onto it
module load Pytorch-gpu
python train.py --smoke-test
```
Exit the shell to end the session. **Don't use this for long training runs** — it's for quick checks only; use batch jobs (Option B) for real training.

### Option B — Batch job (this is what you'll use for actual training runs)
Create a script, e.g. `train_job.sh`:
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2                    # request both A100s on the node; use gpu:1 for one
#SBATCH --time=24:00:00                 # max 4 days (4-00:00:00) per QoS policy
#SBATCH --partition=gpu
#SBATCH --job-name=rl_train
#SBATCH --output=/scratch/24m1531/logs/job.%J.out
#SBATCH --error=/scratch/24m1531/logs/job.%J.err

module purge
module load miniconda
conda activate rl_env

cd /scratch/24m1531/my_project
python train.py --config configs/main.yaml
```

Submit and monitor:
```bash
sbatch train_job.sh
squeue --me                       # check status (PD = pending, R = running)
scontrol show job <jobid>         # detailed info
sacct -j <jobid>                  # resource usage after completion
scancel <jobid>                   # kill it if needed
```

**GPU-specific notes:**
- `--gres=gpu:1` or `--gres=gpu:2` is **mandatory** in the script if you want GPU(s) — omitting it gets you CPU-only allocation even on the gpu partition.
- QoS limits: up to **10 simultaneous jobs**; default walltime is 2 hours if unspecified, so **always set `--time`** explicitly. Max is 4 days per job (longer needs a support ticket).
- `sinfo` shows live partition/node availability:
```bash
sinfo
```

---

## 5. Running DL/RL Training — Practical Workflow

1. **Prototype locally on your Mac** (small model / tiny env) to make sure the code runs end-to-end.
2. **Push code + a small sample of data** to `/scratch/24m1531/` via `scp`/`rsync`.
3. **Smoke-test on an interactive GPU session** (`salloc`, §4 Option A) with a tiny run (few steps/episodes) to catch bugs fast, without waiting in the batch queue.
4. **Submit the full run as a batch job** (§4 Option B). Use `--job-name`, distinct log files, and checkpoint periodically (RL runs especially — save checkpoints so a wall-time cutoff doesn't lose progress).
5. **Checkpoint/save models to `/scratch/24m1531/my_project/checkpoints/`.** For anything precious, periodically `scp` it out to your Mac or to `/home/24m1531/` — scratch is not backed up and has the 3-month purge policy.
6. **Monitor:**
   ```bash
   squeue --me
   tail -f /scratch/24m1531/logs/job.<jobid>.out
   ```
7. **Multi-run / hyperparameter sweeps:** use SLURM job arrays instead of submitting one-by-one:
   ```bash
   sbatch --array=0-9 --gres=gpu:1 train_job.sh
   # inside train_job.sh, use $SLURM_ARRAY_TASK_ID to pick hyperparams/config index
   ```

### Jupyter Notebook on a GPU node (optional, useful for interactive experimentation/visualization)
```bash
# 1. On login node, get a GPU node
salloc --nodes=1 --time=2:00:00 --gres=gpu:1 --partition=gpu
squeue --me                   # note assigned node, e.g. gpu007
ssh gpu007

# 2. On that node
module load miniconda
conda activate rl_env
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
# note the token printed

# 3. From a NEW terminal on your Mac, tunnel in:
ssh -p 4422 -t -t 24m1531@paramrudra.iitb.ac.in -L 8888:localhost:8888 ssh gpu007 -L 8888:localhost:8888

# 4. In your Mac browser:
https://localhost:8888   # paste the token
```

---

## 6. Bringing Trained Models Back to Your Mac

Once training is done, pull down whatever you need (model weights, logs, plots):
```bash
scp -r -P 4422 24m1531@paramrudra.iitb.ac.in:/scratch/24m1531/my_project/checkpoints ~/models/
```
or for ongoing sync while a job runs:
```bash
rsync -avz -e "ssh -p 4422" 24m1531@paramrudra.iitb.ac.in:/scratch/24m1531/my_project/checkpoints/ ~/models/checkpoints/
```

### Setting up your Mac to actually *use* the model (inference/fine-tuning locally)
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for macOS (Apple Silicon or Intel build as appropriate).
2. Recreate a matching environment locally:
   ```bash
   conda create -n rl_env python=3.10 -y
   conda activate rl_env
   pip install torch stable-baselines3 gymnasium   # match versions used on Rudra where possible
   ```
   - For Apple Silicon Macs, PyTorch supports the **MPS backend** (`torch.device("mps")`) instead of CUDA — swap the device string in your inference code.
3. Load your checkpoint and run inference/evaluation as usual — no cluster-specific code should be needed at this stage since training artifacts (model weights) are portable.

**Note on precision/version mismatches:** try to keep the same major PyTorch/TensorFlow version on both ends to avoid checkpoint-loading issues (Rudra uses PyTorch 2.2.x / TensorFlow 2.15.0 as of this manual).

---

## 7. Quick Command Cheat Sheet

| Task | Command |
|---|---|
| Log in | `ssh 24m1531@paramrudra.iitb.ac.in -p 4422` |
| Upload files | `scp -r -P 4422 <local> 24m1531@paramrudra.iitb.ac.in:<remote>` |
| Download files | `scp -r -P 4422 24m1531@paramrudra.iitb.ac.in:<remote> <local>` |
| See partitions | `sinfo` |
| Submit batch job | `sbatch train_job.sh` |
| Check my jobs | `squeue --me` |
| Cancel job | `scancel <jobid>` |
| Interactive GPU shell | `salloc --nodes=1 --time=1:00:00 --gres=gpu:1 --partition=gpu` |
| Job details | `scontrol show job <jobid>` |
| Resource usage after job | `sacct -j <jobid>` |
| Load conda base | `module load miniconda` |
| Load a GPU DL framework | `module load Pytorch-gpu` (or `Tensorflow-gpu`, `Keras-gpu`) |
| Change password | `passwd` (once logged in) |

---

## 8. Housekeeping / Good Habits

- Compile/edit on login node; **only run/execute via SLURM on compute nodes.**
- Never let jobs write output to `/dev/null` — always keep `--output`/`--error` logs for debugging.
- Set realistic `--time` limits — tighter estimates schedule faster (backfill scheduler).
- Back up anything important from `/scratch` to `/home` or your Mac — scratch has no backup and a 3-month auto-delete on untouched files.
- Support/tickets: `https://paramrudra.iitb.ac.in/support` or email `rudrasupport@iitb.ac.in`.
- If publishing results trained on Rudra, acknowledge NSM per the manual's required citation text.
