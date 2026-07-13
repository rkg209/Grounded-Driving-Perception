---
description: Implement exactly ONE task from an approved spec, then stop
argument-hint: <NN-spec-name> [task number]
---

Implement **one** task from **specs/$ARGUMENTS.md**: $ARGUMENTS

## The rules (CLAUDE.md §3)

1. **One task. Then stop.** Not "one task and the obvious next one". Not "while I was in there".
   The stopping is the point: it is what keeps each change reviewable and each result traceable.
2. The spec must be `approved`. If it is not, stop and say so.
3. Reuse what exists (`gdp.config`, `gdp.data`, `gdp.paths.run_dir`, `gdp.seed`). A second config
   loader or a second dataset schema is a bug, not a feature.
4. Write the test **with** the code, not after it. It must fail before the change and pass after.
5. Results go to `runs/<spec>/<ts>/metrics.json` with provenance (config, checkpoint, split, git SHA).
   A number that exists only in the chat log does not exist.
6. **Never launch a full training run.** Emit a SLURM script (`/train-job`). The `guard-bash` hook
   will block you; that is the system working.
7. Do not weaken a test to make it pass. If a test is wrong, say it is wrong and why.

## When the task is done

1. Run the verification: the actual command, the actual output. Do not assume.
2. Tick the checkbox in the spec's `## Tasks`.
3. **Append the `progress_report.md` entry** (`/progress`). The task is **not done** without it —
   including what broke and how you fixed it.
4. Report: what changed, what was verified (with real output), and what the next task is.
5. **Stop.**
