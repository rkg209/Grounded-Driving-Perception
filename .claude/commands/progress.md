---
description: Append an entry to progress_report.md (mandatory after every change)
argument-hint: [what changed]
---

Append an entry to `progress_report.md` for: $ARGUMENTS

## How

1. Read the **end** of `progress_report.md` to get the last `SEQ-` number. Increment it.
2. **Append** — never edit, never reorder, never delete a past entry. The file is the project's
   history, including the parts that went wrong. Those are the parts an interviewer will ask about,
   and the parts that prove the work was real.
3. Use exactly this template (CLAUDE.md §5):

```markdown
## [SEQ-000N] <short title>

**Date:** YYYY-MM-DD · **Spec:** <NN-name or "—"> · **Status:** <done | partial | reverted>

### What
<concretely: files, commands, behaviour. Not "improved the data pipeline" — say what changed.>

### Why
<the reason. Tie it to a spec, an H-item, or a bug. "It was the next task" is not a why —
why is that task in the project at all?>

### How
<the approach, and the alternatives you rejected and why. The rejected paths are the
engineering judgment; without them the entry is a changelog, not a story.>

### Issues & resolutions
<what broke, what the error actually said, what the root cause turned out to be, how you fixed it.
"None." only if genuinely nothing broke — do not sanitise. A build with no problems reads as a
build with no depth.>

### Verification
<the exact commands run and their real output. Not "tests pass" — "34 passed in 1.72s"; not
"exported fine" — the parity tolerance you actually measured.>
```

## Quality bar

The end state is a single file that tells a stranger the **whole story** of how this project was
built: what we did, **why**, **how**, what broke, and how each thing was resolved. Write each entry
as a paragraph of that story, not as a commit message.
