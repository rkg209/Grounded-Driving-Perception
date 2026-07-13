---
description: Draft a new spec in specs/ from the charter, in the project's spec format
argument-hint: <NN-short-name> [one-line objective]
---

Draft a new spec: **$ARGUMENTS**

1. Re-read `grounded-driving-perception-spec.md` (the charter) and `CLAUDE.md` (H1–H9).
2. Write `specs/$ARGUMENTS.md` with exactly these sections:
   - **Objective** — one paragraph, concrete.
   - **Inputs / outputs** — what goes in, what artifacts come out, and *where* (`runs/<spec>/<ts>/`).
   - **Approach** — the plan, including the parts that are easy to get subtly wrong.
   - **Acceptance criteria** — numbered and **falsifiable**. "Works well" is not a criterion.
     Each must be checkable by a command or a test.
   - **Honesty contract** — which of H1–H9 this spec could violate, and *the specific action* that
     would violate them. Be concrete: "reporting INT8 latency without INT8 mAP violates H8."
   - **Out of scope** — what this spec explicitly does not do.
   - **Risks** — what is most likely to go wrong, and what we would do about it.
3. Note the **compute target** (laptop vs cluster, CLAUDE.md §4) and how it stays verifiable on
   `tests/fixtures/` without the real dataset.
4. Add the row to `specs/README.md` with status `draft`.
5. **Stop.** Do not implement. A human approves the spec (`draft` → `approved`) before any code.
