---
description: Break an approved spec's plan into small, independently verifiable tasks
argument-hint: <NN-spec-name>
---

Break the plan for **specs/$ARGUMENTS.md** into tasks.

Rules for a good task here:

- **One task = one coherent change that can be verified on its own.** If verifying it requires a
  second task to be done first, they are one task or the split is wrong.
- Each task states: what changes, **how it is verified** (the exact command or test), and its
  compute target (laptop / cluster).
- **Tests come with the code, not after.** A task that adds a loader adds its test.
- Order tasks so the **cheap sanity checks come first** — the overfit-20-images check, the token-span
  round-trip, the scene-overlap assert. These catch the bugs that would otherwise waste a full
  cluster run.
- Any task requiring a full training run produces a **SLURM script** to hand to the user; it does not
  run training in-session.

Write the task list into the spec file under a `## Tasks` section (checkboxes), so progress is
visible in the repo rather than only in this conversation.

Then **stop**. `/implement` takes exactly one task.
