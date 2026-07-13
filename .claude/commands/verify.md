---
description: Verify the current change actually works — real commands, real output
argument-hint: [NN-spec-name]
---

Verify the current change. **Observe behaviour; do not assume it.**

1. **Baseline gates** (always):
   ```bash
   uv run ruff check .
   uv run pytest
   bash scripts/smoke.sh
   ```
2. **Exercise the actual change**, not just the test suite: run the CLI command it affects, load the
   artifact it produced, print the numbers it wrote. A passing unit test is not evidence that the
   feature works end-to-end.
3. **Check the spec's acceptance criteria one by one** ($ARGUMENTS). For each: state the criterion,
   the command that checks it, and the real result. Any criterion you cannot check → say so; do not
   silently skip it.
4. **Audit the artifacts:**
   - Does `runs/<spec>/<ts>/metrics.json` exist, with provenance (config, checkpoint, split, SHA)?
   - Is every headline metric accompanied by its **baseline on the same split** (H8)?
   - Was any metric computed on `tests/fixtures/` (synthetic)? If so it is a **plumbing check, not a
     result** — label it, and never let it into a table (H7).
5. **Report honestly.** If something failed, say it failed and paste the output. A green summary over
   a red run is the single worst thing you can do in this repo — the whole project's value is that
   its claims are true.

Then append the `progress_report.md` entry (`/progress`) if it is not there yet.
