---
description: Turn an approved spec into an implementation plan (no code)
argument-hint: <NN-spec-name>
---

Plan the implementation of **specs/$ARGUMENTS.md**.

1. Check the spec's status in `specs/README.md`. If it is not `approved`, **stop** and tell the user
   it needs review first (CLAUDE.md §3).
2. Read the spec, the charter, and the existing code. **Find what already exists** — `gdp.config`,
   `gdp.data.load_dataset`, `gdp.paths.run_dir`, `gdp.seed` — and reuse it. Do not invent a second
   config system or a second dataset loader.
3. Produce a plan covering:
   - the files to create/modify, and what each is responsible for;
   - the **verification path on the laptop** (fixtures, no dataset) *and* the real path (cluster);
   - which acceptance criteria each step satisfies;
   - the **subtle failure modes** for this spec (e.g. box format, token spans, label masking,
     scene-level leakage) and how the plan catches them — ideally with a cheap test that fails loudly;
   - which H-items are at risk and where.
4. State the compute target for every step. **If a step needs full training, it becomes a SLURM
   script (`/train-job`), never an in-session run.**
5. **Stop.** No code. Next step is `/tasks`.
