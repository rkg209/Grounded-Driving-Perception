---
name: progress-log
description: Append a well-formed entry to progress_report.md. Use after EVERY change to code, specs, configs, or results — a change is not done until its entry exists.
---

# progress_report.md

One file that tells the **whole story** of how this project was built: what we did, **why**, **how**,
what broke, and how each thing was resolved. At the end of the project it should be readable on its
own, by a stranger, as the complete narrative.

It is also the file a Honda interviewer will find most interesting, because it is the only one that
shows *judgment under uncertainty* rather than a finished artifact.

## Rules

1. **Append after every change.** Code, specs, configs, results — all of them. The change is **not
   done** until the entry exists (CLAUDE.md §5).
2. **Append-only.** Never edit, reorder, or delete a past entry — not even a wrong one. If something
   turned out to be a mistake, write a *new* entry saying so and what replaced it. The wrong turn is
   part of the story and is usually the most instructive part.
3. **One entry per change.** Do not batch a day's work into one blob; the sequence is the point.
4. Sequence numbers increment: `SEQ-0001`, `SEQ-0002`, …

## Template

```markdown
## [SEQ-000N] <short title>

**Date:** YYYY-MM-DD · **Spec:** <NN-name or "—"> · **Status:** <done | partial | reverted>

### What
### Why
### How
### Issues & resolutions
### Verification
```

## What separates a good entry from a changelog

- **What** — concrete. Files, commands, observable behaviour. Not "improved the data pipeline" but
  "added `gdp/data/prompt.py` mapping class index → prompt token span; `gdp data prepare` now emits
  `det_val_coco.json`."
- **Why** — the *reason*, tied to a spec, an H-item, or a bug. **"It was the next task" is not a
  why** — why is that task in the project at all? ("Grounding-DINO scores boxes against text tokens,
  so a class label must be a token span; without this mapping the model trains but detects nothing.")
- **How** — the approach **and the alternatives rejected, with the reason**. This is where the
  engineering judgment lives. An entry with no rejected alternative usually means the decision wasn't
  examined.
- **Issues & resolutions** — what broke, **what the error actually said**, what the root cause turned
  out to be, and the fix. Write "None." only if genuinely nothing broke. **Do not sanitise.** A
  project with no recorded problems reads as a project with no depth — or a dishonest log, which is
  worse.
- **Verification** — the **real** commands and their **real** output. Not "tests pass" but
  "`uv run pytest` → 34 passed in 1.72s". Not "export worked" but the numerical parity tolerance you
  actually measured.

## The test

> Could a stranger read `progress_report.md` end to end and understand not just *what* this project
> is, but *why it is shaped that way, and what it cost to build*?

If yes, the file is doing its job.
