# Progress Report — Grounded Driving Perception

The story of how this project was built: what we did, **why**, **how**, what broke, and how each
thing was resolved. **Append-only** — entries are never edited or deleted, including the wrong turns.
The wrong turns are the most instructive part, and the part an interviewer will ask about.

Maintained per `CLAUDE.md` §5 and the `progress-log` skill. A change is not done until it appears here.

---

## [SEQ-0001] Project bootstrap — SDD repository, specs, and Claude Code tooling

**Date:** 2026-07-13 · **Spec:** — (bootstrap) · **Status:** done

### What

Turned a single narrative charter (`grounded-driving-perception-spec.md`) into a working
spec-driven-development repository:

- `CLAUDE.md` — the agent constitution: project identity, the **Honesty Register (H1–H8)**, the SDD
  loop, the laptop-vs-cluster rule, and the mandate that this file be updated after every change.
- `specs/` — the backlog decomposed into 11 specs (00–10) plus a status index. Each carries
  falsifiable **acceptance criteria** and an explicit **honesty contract** naming the H-items it
  could violate.
- `.claude/` — 8 slash commands (`/new-spec`, `/plan`, `/tasks`, `/implement`, `/verify`,
  `/progress`, `/claims-check`, `/train-job`), 5 skills (`detection-eval`, `onnx-export`,
  `slurm-job`, `honesty-audit`, `progress-log`), 1 sub-agent (`claims-auditor`), and 4 hooks wired
  in `settings.json`.
- `README.md`, `progress_report.md`, `pyproject.toml`, `.gitignore`, `Makefile`.

### Why

The charter is prose: it states *what* to build and — crucially — *what not to claim*, but it is not
executable and its guardrails are unenforceable as written. Two things needed to become mechanical:

1. **The honesty guardrails.** The charter's value proposition is that every claim is defensible to a
   Honda reviewer. Guardrails that live only in a paragraph get forgotten at 1 a.m. on the weekend the
   fine-tune finally works. So they were distilled into a numbered register (H1–H8), embedded in the
   constitution, made a required section of every spec, and given an adversarial auditor agent.
2. **The compute reality.** The dev machine is an M4 with 16 GB. It *cannot* fine-tune Grounding-DINO
   or a 3B VLM — not slowly, not at all. Every spec therefore had to be designed around a
   laptop/cluster split from the start, rather than discovering it at spec 03.

### How

**Model decisions (locked with the user):** Grounding-DINO (`IDEA-Research/grounding-dino-tiny`) for
Stage 1; Qwen2.5-VL-3B + DriveLM for Stage 2; IITB SLURM cluster for training; datasets not yet
downloaded.

**Before committing to Grounding-DINO, two load-bearing facts were verified against the HF docs
rather than assumed** — both had the power to invalidate the whole Stage-1 plan:

- *Is it even fine-tunable in `transformers`?* Historically `GroundingDinoForObjectDetection` shipped
  inference-only, with no loss. It now **accepts `labels`** (`list[dict]` with `class_labels` +
  `boxes`) and computes the bipartite-matching loss. **Consequence: no MMDetection dependency** — a
  significant simplification that was not safe to assume.
- *Can it be exported to ONNX?* Its custom multi-scale deformable-attention kernels are **not**
  exportable — but `GroundingDinoConfig.disable_custom_kernels=True` swaps in a pure-PyTorch path,
  and the docs state this flag is *"necessary for the ONNX export"*. This de-risked spec 05, the
  project's checkpoint milestone. The flag is already surfaced in `gdp.config.DetectorConfig`.

**Rejected alternatives:**

- *OWLv2 instead of Grounding-DINO* — easier to export, weaker on the referring phrases that are the
  entire point of spec 04. Kept as a documented fallback, not the primary.
- *Enforcing honesty by review alone* — replaced with mechanism: an auditor agent, a `/claims-check`
  command, and a per-spec honesty contract.
- *Hoping the agent remembers to update this file* — replaced with a **Stop hook** that blocks the
  session from ending while `progress_report.md` is older than the code it describes. Documentation
  discipline that depends on goodwill decays; this one cannot.

**The keystone design choice — the synthetic fixture.** BDD100K needs registration and is ~7 GB;
nuScenes/DriveLM likewise. Without data, nothing is testable, and the project stalls before it starts.
So `scripts/make_fixtures.py` draws four crude road scenes (sky, road, lane line, coloured boxes for
car/bus/pedestrian/traffic-light) whose ground truth is *exact because we drew it*, and writes them as
COCO-style annotations. The whole repo — tests, CLI, config, and later the ONNX export — runs on the
laptop with **zero downloads and zero GPU**. Every later spec is required to keep a fixture path so it
stays verifiable offline. The images are watermarked "NOT REAL DATA" and the annotations JSON carries
a `NOT real BDD100K` description, with a test asserting it: a synthetic fixture that could be mistaken
for real data would be an H7 violation waiting to happen.

### Issues & resolutions

1. **`uv sync` failed: `OSError: Readme file does not exist: README.md`.** `pyproject.toml` declared
   `readme = "README.md"` before that file was written, so the package build backend refused to build.
   Wrote `README.md` first; sync then resolved cleanly. Trivial, but a good argument for running the
   toolchain early rather than writing twenty files and hoping.

2. **A real bug, caught by reasoning rather than by a test — `from __future__ import annotations`
   silently breaks dataclass introspection.** `config.py` built nested config objects by checking
   `is_dataclass(field.type)`. With PEP 563 postponed annotations (which that import enables),
   `Field.type` is the **string** `"PathsConfig"`, not the class — so `is_dataclass()` is `False` for
   every nested field, and `paths:`/`detector:`/`vlm:` sections would have been passed through as raw
   `dict`s. The config would have *loaded without error* and then failed much later with an
   `AttributeError: 'dict' object has no attribute 'model_id'`, in a cluster job, at hour six.
   **Fix:** resolve field types with `typing.get_type_hints(cls)` instead of reading `Field.type`.
   The `test_merge_is_deep_and_right_wins` and `test_bdd100k_overlay_only_changes_dataset` tests now
   pin this behaviour.

3. **Writing `.claude/settings.json` was blocked by the permission system.** The first draft bundled a
   `permissions.allow` list (wildcards like `Bash(uv run gdp:*)`, `Bash(make:*)`) alongside the hooks.
   Pre-approving future shell commands was never requested, and the classifier was right to stop it.
   **Fix:** settings.json now contains hooks only. Permission grants, if wanted, are the user's call.

### Verification

```
uv sync                                    → resolved (torch 2.13.0, typer 0.26.8)
uv run python scripts/make_fixtures.py     → wrote 4 images + 14 boxes
uv run pytest                              → 34 passed in 1.72s
uv run ruff check .                        → All checks passed!
bash scripts/smoke.sh                      → SMOKE OK (device: mps)
```

Guard hook tested against five commands — blocks (exit 2) training launches, `git add data/`, and
recursive deletes; allows (exit 0) `uv run pytest` and ordinary `git add src/...`:

```
uv run gdp train ...      → BLOCKED (exit 2)  "use /train-job"
git add data/bdd100k      → BLOCKED (exit 2)  "not redistributable"
rm -rf ./runs             → BLOCKED (exit 2)
uv run pytest             → allowed (exit 0)
git add src/gdp/cli.py    → allowed (exit 0)
```

Fixture `scene_000.jpg` was rendered and visually inspected: sky, road, lane line, a traffic light
above the horizon, vehicles on the road surface, "NOT REAL DATA" watermark present.

---

## [SEQ-0002] Spec 00 — minimum working codebase

**Date:** 2026-07-13 · **Spec:** 00-scaffold · **Status:** done

### What

The `gdp` package, importable and tested, with nothing that pretends to work:

- `src/gdp/config.py` — YAML → dataclass, deep-merged left-to-right, validated, **unknown keys
  rejected**. Holds `BDD100K_CLASSES` (order is load-bearing — it defines class indices).
- `src/gdp/paths.py` — repo-root-relative resolution + `run_dir()` for timestamped outputs.
- `src/gdp/seed.py` — determinism across python/numpy/torch, and device selection `cuda → mps → cpu`.
- `src/gdp/data.py` — COCO-style loader (`Box`, `Sample`, `Dataset`) with validation.
- `src/gdp/cli.py` — the full command surface (`info`, `data`, `detect`, `evaluate`, `deploy`, `vqa`,
  `demo`); only `info` runs.
- `configs/`, `tests/` (34 tests), `scripts/smoke.sh`, `scripts/make_fixtures.py`.

### Why

Every later spec needs somewhere to land and a way to be checked. Three choices here are worth the
words, because each prevents a specific, expensive failure later:

- **The CLI is an honest roadmap.** All seven commands exist from day one, but unimplemented ones exit
  2 and name the spec that will implement them (`gdp detect` → "specs/02-zeroshot-baseline.md"). So
  `gdp --help` reports the project's *true* progress and structurally cannot overstate it. A skeleton
  that silently no-ops, or worse returns empty results, is how a demo accidentally becomes a lie.
- **Unknown config keys are a hard error.** A silently-ignored `box_treshold` typo means the run used
  0.25 while the report says 0.5 — and the number is now fiction that nobody can trace. Better to fail
  at load.
- **`select_device` refuses to fall back.** Asking for `cuda` when CUDA is missing raises instead of
  quietly using CPU. Silently training on CPU for ten hours on a cluster node is a failure mode worth
  being loud about.

### How

Standard `src/` layout, `uv` for dependency management, `typer` for the CLI, `ruff` for lint+format,
`hatchling` as the build backend. Tests cover config merge/validation, seed determinism, device
selection, the fixture data contract, and CLI behaviour — including a parametrised test asserting that
**every** unimplemented command exits 2 and names its spec, so the roadmap cannot silently drift from
the spec backlog.

Deliberately **not** built: any model code, any dataset download, any training, any ONNX. Spec 00 is
scaffolding, and scope creep here would have meant writing Grounding-DINO code before spec 01 defined
the data contract it consumes.

### Issues & resolutions

Covered in SEQ-0001 (items 1 and 2 — the `README.md` build failure and the `get_type_hints` bug — both
surfaced while building this codebase). Nothing further broke.

One thing worth recording that did **not** break but was close: `torch` resolved to 2.13.0 in the uv
venv while the system had 2.8.0. Harmless now, but the cluster environment will differ again, so
`metrics.json` provenance must record the torch version alongside the git SHA — otherwise a
reproducibility gap opens the first time a result is questioned.

### Verification

Every acceptance criterion in `specs/00-scaffold.md`, checked:

| # | Criterion | Result |
|---|-----------|--------|
| 1 | `uv sync` resolves; `pytest` green | **34 passed in 1.72s** |
| 2 | `gdp info` emits valid JSON with a real device | `{"device": "mps", "seed": 42, "num_classes": 10, ...}` |
| 3 | Unimplemented commands exit 2 naming their spec | 6/6, asserted by parametrised test |
| 4 | `scripts/smoke.sh` exits 0 | **SMOKE OK** (5/5 stages) |
| 5 | `ruff check .` clean | **All checks passed!** |
| 6 | Typo'd config key raises | `ValueError: unknown config key(s) in detector: ['box_treshold']` |
| 7 | `set_seed` reproducible | asserted across python/numpy/torch |
| 8 | No file claims a metric | confirmed — the repo contains no results yet, by construction |

**Next:** spec 01 (`01-data-bdd100k`) — but it is `draft`. It needs human approval before any code
(CLAUDE.md §3), and BDD100K registration needs to be started, since that is the long-lead item.

---

## [SEQ-0003] Scope review against the "Grounded Driving Intelligence System" proposal

**Date:** 2026-07-14 · **Spec:** — (cross-cutting) · **Status:** done

### What

Evaluated a revised project document (proposing six modules: open-vocab detection, grounding, VLM QA,
**temporal understanding**, **risk assessment**, **scene report generation**, plus deployment) against
the existing charter and spec backlog. Adopted three of its ideas, reframed two, and rejected two —
then updated the register, five specs, the README, the auditor, and the honesty skill accordingly.

- **Adopted as-is:** precision/recall alongside mAP (spec 02); **memory footprint** — model size on
  disk + peak RSS — in the deployment benchmark (spec 05); the **scene report generator** (spec 09).
- **Reframed:** **risk assessment** → measured via DriveLM's official planning/behaviour/safety
  categories in spec 08, *not* a hand-rolled engine.
- **Deferred:** **temporal reasoning** → new **spec 11**, optional stretch, demo-only, explicitly the
  first thing cut.
- **Rejected:** a rule-based risk engine with a "risk detection accuracy" metric; any temporal
  accuracy number.
- **New register rule — H9:** *no self-invented benchmark, no self-invented metric.*
- **Fixed H5**, which was wrong (see below).

### Why

The document is a genuine improvement in *framing* — "Grounded Driving Intelligence System" tells a
better story than "grounded perception", and explainable scene reports are a real differentiator. But
two of its modules would have quietly destroyed the project's only real asset: that every claim
survives scrutiny.

**Why the risk engine was rejected.** It proposed a rule-based hazard estimator using "distance
thresholds" and "relative positions", scored by "Risk Detection Accuracy". Two independent
show-stoppers:

1. **BDD100K is monocular with no camera calibration and no depth ground truth.** "Pedestrian close to
   the lane boundary" is therefore not computable in metres — only in *pixels*, which is not distance.
   A hazard rule built on pixel proximity is a demo wearing the costume of physics, and the first
   interviewer to ask "close in what units?" ends the conversation.
2. **There is no ground truth for "risk".** The accuracy would be measured against labels we invented,
   using rules we wrote. That is circular — it measures our agreement with ourselves. It is precisely
   the "unfalsifiable" failure the charter calls out in competing projects (§6: *"no fine-tuning, no
   benchmark, no metrics — unfalsifiable"*).

**But risk assessment did not need to be cut — only relocated.** DriveLM's official taxonomy already
contains prediction / planning / behaviour questions ("is it safe to merge left?"). Those *are* risk
reasoning, and they arrive with official ground truth and an official metric. So the capability is
delivered **measured**, in spec 08, at **zero additional cost** — we were already computing
per-category accuracy; we simply had not named the planning/behaviour rows "risk". This is strictly
better than the proposal: the same headline, backed by a benchmark instead of by our own opinion.

**Why temporal was deferred rather than adopted.** "Is the pedestrian approaching the road?",
"trajectory trend", "hazard indicators" — this is trajectory prediction re-entering through the back
door, i.e. the exact project that was deliberately abandoned in favour of this one. It also has no
official benchmark (DriveLM and nuScenes-QA are keyframe-based), so any metric would again be
self-invented. And it is the fastest way to overrun the schedule and miss the **spec-05 checkpoint**,
which is the thing that makes the project defensible *at all*.

There is an honest sliver, and spec 11 captures it: **Qwen2.5-VL natively accepts multiple images**, so
consecutive frames can be fed to the already-fine-tuned model with **no tracker and no new
architecture**. That is a legitimate qualitative demo. It is not a result, and spec 11 forbids it from
ever carrying a number.

### How

**The most valuable thing this review produced was a correction to our own register.** H5 previously
read *"No tracking, no trajectory prediction, no planning/control."* Read literally, that would have
**forbidden answering DriveLM's own planning questions** — yet the charter itself lists *"Is it safe to
merge left?"* as a flagship Stage-2 example. The rule was conflating two different things. H5 now
distinguishes them explicitly:

> **the module vs. the question** — we ship no tracker, no path predictor, no planner, and emit no
> driving commands; but *answering a benchmark's planning question in natural language is a VQA task,
> not a planner.*

Left unfixed, this would have surfaced as a contradiction midway through spec 08, with the register
appearing to forbid the project's own headline result.

**H9 was added** to generalise what H2 only said about VQA: *if a capability has no official ground
truth, it gets no accuracy number.* This is the rule that makes the risk-engine and temporal-metric
rejections principled rather than ad hoc, and it will keep saying no long after this conversation is
forgotten. The spec-04 grounding set is named as the single sanctioned exception — permitted **only**
because its construction, bias, and drop-rate are fully documented, and still labelled self-built
wherever it appears.

**Scene report — the H6 design problem.** Composing detector boxes *and* VLM answers into one summary
is the most natural way to imply a single joint system, which H6 exists to prevent. Resolved by
requiring **per-line model attribution** in the report (`[detector — Stage 1]` / `[VLM — Stage 2]`),
with an acceptance test asserting no unattributed sentence can be emitted. The report is thereby a
*presentation layer over two models*, and the UI never lets a viewer forget it.

**Rejected alternatives:** adding a `10-risk-engine` spec (would have needed depth we don't have);
scoring the scene report with DriveLM's language metrics (the metric only partially fits the task, and
a bad-fit metric is worse than an honest "unmeasured"); promoting temporal into core scope (schedule
risk against the checkpoint, with no claimable output at the end of it).

### Issues & resolutions

1. **A latent self-contradiction in our own honesty register, caught only because the new document
   pushed on it.** H5 as written forbade planning; the charter's own Stage-2 example asks a planning
   question. Nobody had noticed because no spec had yet had to reconcile them. Fixed by rewriting H5
   around the module/question distinction. *This is the argument for reviewing incoming ideas against
   the register rather than waving them through — the register got audited too.*

2. **`H1–H8` was hard-coded in nine places** across `CLAUDE.md`, three specs, two commands, the skill,
   the agent, and the session-start hook. Adding H9 meant they all silently became wrong — the kind of
   drift where the constitution says nine rules and the auditor checks eight. Swept them all to
   `H1–H9` and grepped to confirm zero stragglers remain.

### Verification

```
grep -rn "H1–H8|H1-H8"        → no stale references remain
uv run ruff check .           → All checks passed!
uv run pytest                 → 34 passed
bash scripts/smoke.sh         → SMOKE OK (device: mps)
```

Docs-and-specs change only; no source code touched, so the test suite is unchanged and green as a
regression check rather than as new evidence.

Register now H1–H9; backlog now 00–11 (11 = optional stretch). Risk assessment has a home (spec 08,
measured), the scene report has a home (spec 09, attributed + labelled), and temporal has a home
(spec 11, demo-only, first to cut). **No new claim was created that the project cannot defend.**

**Next:** unchanged — spec 01 needs approval, and BDD100K/nuScenes registration is still the long-lead
item blocking everything downstream.

---

## [SEQ-0004] Ban `Co-Authored-By` trailers; purge them from git history

**Date:** 2026-07-14 · **Spec:** — (tooling) · **Status:** done

### What

- Added **CLAUDE.md §7 "Git rules"**: never add a `Co-Authored-By:` trailer to a commit message —
  explicitly overriding any default instruction to do so. Sections renumbered (Conventions → §8,
  Before you say "done" → §9), and the four stale `§7`/`§8` cross-references in `specs/README.md`,
  `specs/02-zeroshot-baseline.md`, `src/gdp/paths.py` and `src/gdp/config.py` were fixed to match.
- Added a **fourth `guard-bash` rule** that blocks any Bash command containing `Co-Authored-By`,
  so the ban is mechanical rather than a matter of memory.
- **Rewrote both existing commits** to strip the trailer they already carried.

### Why

The trailer was breaking the user's pushes to GitHub. Two separate problems needed solving, and only
one of them was the history:

1. **The existing history was already contaminated** — both commits (`ec107cf`, `b129158`) ended with
   `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Since a push is what fails, a rule that
   only governs *future* commits would have left the repo unpushable anyway.
2. **The trailer is added by a default instruction**, not by choice. A rule in CLAUDE.md is necessary
   but, being a default I am told to follow elsewhere, it is exactly the kind of thing that quietly
   reasserts itself twenty commits from now. So it is also enforced by a hook that inspects the
   command before it runs.

The user also asked to confirm `progress_report.md` exists and that CLAUDE.md mandates maintaining it.
**Both were already in place** from SEQ-0001 — the file has entries SEQ-0001 through SEQ-0003, and
CLAUDE.md §5 makes an entry mandatory after every change (with a Stop hook that blocks the session
while the file is stale). Nothing to add there; this entry is itself the demonstration.

### How

`git filter-branch -f --msg-filter 'grep -v -i "^Co-Authored-By:"' -- --all`, safe because the repo has
**no remote configured**, so no published history could be broken. Both commits were rewritten
(`ec107cf → ceccad6`, `b129158 → b294d93`); the contents are byte-identical, only the messages changed.

**The part that was nearly missed:** after `filter-branch` reported success, `git log --all` *still*
found the trailer. `filter-branch` preserves the pre-rewrite commits under `refs/original/` as a
backup, and those refs are pushable — so the repo would have carried the old commits to GitHub and the
push would have failed anyway, with the branch itself looking perfectly clean. Deleted the backup refs,
expired the reflog, and ran `git gc --prune=now`; only then did the objects actually disappear. *A
history rewrite is not finished when the branch looks right — it is finished when `--all --reflog`
comes back empty.*

**Rejected alternatives:** `git reset --soft` + recommit (loses the authored timestamps and is
fiddlier for more than one commit); leaving history alone and only fixing future commits (would not
have fixed the push, which was the actual complaint).

### Issues & resolutions

1. **`filter-branch` "succeeded" while the trailer was still reachable.** Cause: `refs/original/`
   backup refs, plus reflog entries, still pointed at the original commits. Resolved by deleting those
   refs, expiring the reflog, and garbage-collecting with `--prune=now`. Verified with
   `git log --all --reflog --format='%B' | grep -i co-authored-by` returning **nothing**.

2. **Renumbering CLAUDE.md silently broke four cross-references.** Inserting the new §7 pushed
   Conventions from §7 to §8, so `paths.py` and `config.py` docstrings and two specs were now citing
   the wrong section — the sort of rot that makes a constitution untrustworthy precisely because each
   individual instance is too small to notice. Grepped for every `CLAUDE.md §N` reference and fixed
   all four.

### Verification

```
git log --all --reflog | grep -i "co-authored-by"   → no output (CLEAN)
git log --oneline                                   → b294d93, ceccad6 (both messages trailer-free)
git remote -v                                       → none (rewrite was safe)

guard-bash: git commit ... Co-Authored-By ...       → BLOCKED (exit 2)
guard-bash: git commit -m "fix thing"               → allowed (exit 0)

uv run ruff check .                                 → All checks passed!
uv run pytest                                       → 34 passed
bash scripts/smoke.sh                               → SMOKE OK
```

The ban is now enforced in three places: the constitution (CLAUDE.md §7), the hook (`guard-bash.sh`
rule 4), and the history (purged). **This commit is the first one written under the new rule.**

---

## [SEQ-0005] Back-fill the missing plan artifact for spec 00

**Date:** 2026-07-15 · **Spec:** 00-scaffold · **Status:** done

### What

Added `.claude/plans/00-scaffold.md` — an implementation plan for spec 00, written retroactively
against the scaffold **as actually built**. It records: the files and the design decision each one
encodes; the CLI-as-roadmap and synthetic-fixture ideas that specs 01+ depend on; a mapping of all
eight acceptance criteria to what discharges them; the six subtle failure modes the scaffold guards
against and the test that catches each; the H-items at risk; and an explicit handoff section listing
what spec 01 inherits.

This also creates the `.claude/plans/` directory, which did not previously exist. **No code changed;
spec 00 was and remains `done`, so `specs/README.md` needed no update.**

### Why

CLAUDE.md §3 defines the SDD loop as `/new-spec → /plan → /tasks → /implement → /verify →
/progress`, but spec 00 was built without its plan step ever being persisted — the repo had no
`.claude/plans/` at all. Specs 01–11 will each produce one, so without this the written plan record
starts at 01, and the *keystone* spec is the one with no design document. Spec 00 is where the config
system, the repo-root path contract, the fixture, and the CLI-as-roadmap are decided, and every later
spec inherits all four. Those contracts deserved to be stated somewhere other than in the code that
implements them.

The user asked for a plan for spec 00. Since 00 is already done, a *forward-looking* plan would have
been planning work that had already shipped, so I checked first and asked which of three things was
actually wanted (retroactive record / plan 01 instead / audit-and-re-plan 00). The user chose the
retroactive record.

### How

Re-ran spec 00's full acceptance path **before** writing a word, so the document describes the repo
that exists rather than the one I remembered: 34 tests pass, ruff clean, `gdp info` reports
`"device": "mps"`, smoke exits 0. Then read every source file in `src/gdp/`, both configs, the
fixture generator, and the test names, and wrote the plan from that.

Two things I deliberately did **not** do:

- **Did not write it as if planning from scratch.** A retroactive plan that pretends to be
  prospective is a small lie in a repo whose entire value proposition is that its claims are
  defensible. The document opens by stating plainly that it was written after the fact and verified
  against the code.
- **Did not touch spec 00's status or code.** The temptation with a back-filled artifact is to
  "improve" the thing you are documenting. Documentation and change are separate commits.

The most useful section turned out to be §5 (subtle failure modes), because it captures reasoning
that exists in the code only as a one-line comment — in particular the `from __future__ import
annotations` trap in `config.py`, where `Field.type` is the *string* `"PathsConfig"` so
`is_dataclass()` is always false and nested config sections would silently become raw dicts. That is
a bug someone reintroduces in six months. Now it is written down where a plan-reader will see it.

### Issues & resolutions

**The premise of the request did not hold.** The task was "read spec 00, generate an implementation
plan", but spec 00 is `done` — a fact the session-start hook prints on every launch. Producing a
confident implementation plan for already-shipped work would have been a plausible-looking but
useless artifact. Resolved by verifying the state first and asking the user which of three readings
they meant, rather than guessing.

Nothing broke. The change adds one markdown file and touches no code.

### Verification

```
uv run pytest                  → 34 passed in 0.89s
uv run ruff check .            → All checks passed!
bash scripts/smoke.sh          → exit 0 (SMOKE OK)
uv run pytest -k "unknown_key or synthetic or degenerate or falling_back
                  or missing_image or unknown_category"
                               → 6 passed   (every test named in §5 exists and passes)
git status --short             → ?? .claude/plans/   (nothing else touched)
```

The targeted pytest run is the one that matters: it confirms each test the plan *claims* guards a
failure mode is real and green, so the document cannot be quietly describing tests that do not exist.

## [SEQ-0006] Implementation plan for spec 01 (BDD100K data pipeline)

**Date:** 2026-07-15 · **Spec:** 01-data-bdd100k · **Status:** done

### What

Wrote `.claude/plans/01-data-bdd100k.md` — the implementation plan for spec 01, following the
`/plan` step of the SDD loop. No code, no config, no fixture changes yet; the plan is the artifact.

It specifies: the `gdp/data/` package refactor (with an `__init__` re-export so spec 00's
`from gdp.data import load_dataset` keeps working verbatim), `prompt.py` (the class-index →
token-span mapping), `bdd100k.py` (raw `det_20` → our COCO schema, with drop accounting),
`stats.py` (the sanity report), the `gdp data prepare` sub-command, and an 8-task order.

### Why

CLAUDE.md §3: no production code without an approved spec, and the spec is turned into a plan before
any task is implemented. Spec 00's plan (SEQ-0005, §7 "What spec 01 inherits") explicitly hands the
prompt-span trap to spec 01; this plan is where that hand-off gets a design.

### How

Read spec 01 against the code that already exists (`gdp/data.py`, `gdp/config.py`, `gdp/cli.py`,
`configs/bdd100k.yaml`, the fixture) rather than planning in the abstract. Two facts surfaced that
changed the plan, and neither was visible from the spec alone:

1. **`transformers` is not a dependency and there is no HF cache.** The span mapping — the whole
   point of spec 01 — has no tokenizer to work with. It must be added.
2. **The fixture draws only 4 of the 10 classes** (`rider`, `truck`, `train`, `motorcycle`,
   `bicycle`, `traffic sign` have zero boxes). So spec 01's acceptance criteria 4 ("counts non-zero
   for all 10") and 6 ("exercised on the fixture") *contradict each other* against the fixture as
   built. Caught at plan time rather than at implement time.

Three forks were put to the user rather than assumed:

- **Tokenizer in tests** → HF cache + a committed golden `prompt_spans.json`, test skips loudly if
  the tokenizer is unavailable. *Rejected:* vendoring ~1 MB of tokenizer files into `tests/fixtures/`
  (always-offline, but a third-party artifact in the repo), and hard-failing with no offline path.
  The golden file is what catches tokenizer-version drift — otherwise a bumped tokenizer could shift
  a span and silently change every training label in spec 03 with no test going red.
- **The 4-vs-10-class fixture** → extend `make_fixtures.py` to draw all 10, so the all-10 assertion
  runs on the laptop. *Rejected:* leaving the fixture and only shape-testing the report, which would
  mean the criterion is never verified offline.
- **`positive_map`** (num_classes × num_tokens, what spec 03's loss actually eats) → build it in 01.
  *Rejected:* deferring to 03, which would leave the highest-risk artifact untested until the spec
  that can only be run on the cluster.

Design decisions worth recording: the converter **counts dropped boxes by reason** (unknown
category, degenerate, out-of-frame, no `box2d`) and clips edge-straddling boxes rather than dropping
them; frames with `labels: null` **keep their image entry with zero boxes**, because silently
dropping empty images would shrink the val split that every later mAP is computed on (H8); and the
raw-format fixture gives the converter an offline input, so `convert(raw) == committed
annotations.json` becomes a real parity test between two independent code paths.

### Issues & resolutions

The 4-vs-10-class fixture contradiction above — spec 01 cannot satisfy criteria 4 and 6 as written
without changing the fixture. Resolved by extending the fixture (agreed with the user) rather than
weakening the criterion.

Also noted for implementation: on *real* BDD, `train` may genuinely be zero or near-zero in val, so
"all 10 non-zero" can only ever be a **fixture** assertion. On real data the report flags a class
below `min_boxes_for_eval` as too rare to evaluate (H7) instead of dropping it to flatter later mAP.

### Verification

```
cp → .claude/plans/01-data-bdd100k.md   (223 lines, alongside 00-scaffold.md)
git status --short                      → M progress_report.md, ?? .claude/plans/
```

No code, config, or fixture was touched, so no test run is claimed. The plan's own verification
section (§10) is the command list the implementation will be held to, and it maps each of spec 01's
six acceptance criteria to the specific test that discharges it.

---

## [SEQ-0007] Spec 01 approved; task list written; task 1 — `transformers` dep + `gdp/data/` package refactor

**Date:** 2026-07-15 · **Spec:** 01-data-bdd100k · **Status:** done

### What

Three things, in order:

1. **Marked spec 01 `approved`** in both `specs/01-data-bdd100k.md` and `specs/README.md`. It had
   been sitting at `draft` since SEQ-0006 wrote its plan; CLAUDE.md §3 forbids implementing a
   non-approved spec, so this was confirmed with the user before any code was touched.
2. **Wrote a `## Tasks` section into `specs/01-data-bdd100k.md`**, turning the plan's §8 task order
   into 8 checkbox tasks, each with its own verification command and compute target (all laptop,
   all fixture-only).
3. **Implemented task 1**: `git mv src/gdp/data.py src/gdp/data/core.py`, added
   `src/gdp/data/__init__.py` re-exporting `Box`, `Dataset`, `Sample`, `load_dataset`, and added
   `transformers>=4.44` to `pyproject.toml` dependencies.

### Why

Spec 01's real work — the class-index → prompt-token-span mapping in task 2 — needs a real
tokenizer, and `transformers` was not yet a dependency. Doing the package split as its own first task
(rather than folding it into the `prompt.py` task) keeps each task independently verifiable per
CLAUDE.md §3 rule 1: the refactor's only claim is "spec 00's import path and tests still work", which
is a clean, narrow thing to verify before any new logic is added on top.

### How

Reused the existing `gdp.paths.resolve` import inside `core.py` unchanged — no logic was touched, only
the file's location. The re-export module is intentionally minimal (four names, no logic) so
`from gdp.data import load_dataset` continues to work verbatim for every existing caller
(`gdp/cli.py`, `tests/test_fixtures.py`), which was the explicit constraint from the plan (§2).

**Rejected alternative:** folding the `transformers` dependency add into task 2 instead of task 1.
Rejected because `uv sync` resolving a new dependency is itself a thing that can fail (network,
version conflict) independent of any code being correct, so it deserves its own isolated verification
step before `prompt.py`'s logic is on the table.

### Issues & resolutions

None. `uv sync` resolved `transformers>=4.44` to `5.13.1` cleanly (pulling in `tokenizers`,
`huggingface-hub`, `safetensors` as expected transitive deps), and the package split needed no source
changes beyond the new `__init__.py`.

### Verification

```
uv sync                                              → resolved; transformers==5.13.1 installed
uv run pytest                                        → 34 passed in 0.98s (spec 00 suite untouched)
uv run ruff check .                                  → All checks passed!
uv run python -c "from gdp.data import Box, Dataset, Sample, load_dataset; print('ok')"
                                                      → ok
git status --short                                   → R  src/gdp/data.py -> src/gdp/data/core.py
                                                        (plus the __init__.py addition, pyproject.toml,
                                                         uv.lock, and the two spec/README status edits)
```

**Next:** task 2 — `prompt.py` + golden file + `tests/test_prompt.py`, the spec's real risk.

---

## [SEQ-0008] Task 2 — `prompt.py`, the class-index -> prompt-token-span mapping

**Date:** 2026-07-15 · **Spec:** 01-data-bdd100k · **Status:** done

### What

- `src/gdp/data/prompt.py` — `PromptSpans`, built via `from_classes(classes, tokenizer=None)`:
  constructs the prompt and char spans with a running cursor (never `str.find`), gets token spans
  from the tokenizer's `return_offsets_mapping=True` (never by counting words), and asserts every
  span is non-empty, disjoint, and strictly increasing at construction time. Also exposes
  `token_class_ids()`, `positive_map(max_text_len)` (L1-normalized rows), `decode_span()`, and
  `to_json()`. `get_tokenizer()` loads and process-caches `bert-base-uncased`.
- `tests/fixtures/prompt_spans_golden.json` — committed golden output for the real 10-class BDD100K
  prompt.
- `tests/test_prompt.py` — 19 tests: round-trip decode for all 10 classes (including the two
  multi-word classes), the shared-prefix case (`traffic light` vs `traffic sign`), disjoint/
  increasing spans, special tokens owning no class, `positive_map` normalization and its
  `max_text_len` guard, both invariant-violation cases raising, and a golden-file equality check.
  The tokenizer fixture skips loudly (`pytest.skip`, not a silent pass) if the tokenizer can't be
  obtained.

### Why

This mapping is the spec's real risk (per the plan, §3): Grounding-DINO has no class head, so a
class is a token span, and a wrong span produces a model that trains without error and detects
nothing. Doing this before the converter (task 4) means the highest-risk artifact is tested before
anything is built on top of it.

### How

Verified the tokenizer's actual offset-mapping output by hand first (`bert-base-uncased` on the
real 10-class prompt) rather than assuming its shape — confirmed `[CLS]`/`[SEP]` get `(0, 0)`
offsets and multi-word classes like "traffic light" tokenize as `["traffic", "light"]`, both of
which the implementation depends on.

**Rejected alternative:** computing char spans with `str.find(name)`. Rejected per the plan's own
reasoning — `"traffic sign".find("traffic")` would find the wrong occurrence relative to
`"traffic light"` if the classes were ever reordered or renamed, whereas the cursor approach can't
misfire regardless of class names.

### Issues & resolutions

None during implementation. One `ruff` E501 (a line exceeding 100 chars in the new test file) was
caught by the ordinary `ruff check .` pass and fixed by wrapping the string literal — not a design
issue, just a formatting one.

### Verification

```
uv run pytest tests/test_prompt.py -v   → 19 passed in 3.24s
uv run pytest                           → 53 passed in 2.66s (34 spec-00 + 19 new)
uv run ruff check .                     → All checks passed! (after fixing one E501)
```

Manual round-trip check before writing tests: `decode_span(i)` for all 10 `BDD100K_CLASSES` returns
exactly the class name, confirmed by hand against a printed list, including
`"traffic light" -> "traffic light"` and `"traffic sign" -> "traffic sign"`.

**Next:** task 3 — extend `scripts/make_fixtures.py` to draw all 10 classes plus emit raw/dirty
BDD-format fixtures.

---

## [SEQ-0009] Task 3 — fixture extension: all 10 classes + raw + dirty BDD-format fixtures

**Date:** 2026-07-15 · **Spec:** 01-data-bdd100k · **Status:** done

### What

- `scripts/make_fixtures.py` now draws **all 10** `BDD100K_CLASSES` (previously 4: `car`,
  `pedestrian`, `bus`, `traffic light`) via a deterministic per-scene class assignment
  (`SCENE_CLASSES`) rather than random sampling, so every class is guaranteed to appear at least
  once across the 4 scenes — a random `rng.sample` could miss one by chance, which would make
  acceptance criterion 4 ("counts non-zero for all 10") untestable offline.
- Emits `tests/fixtures/mini_bdd_raw/det_val.json` — the same 10 boxes, in raw BDD `det_20` format
  (`box2d` xyxy, one frame per image) — an independent code path's output that the converter (task
  4) must reproduce exactly.
- Emits `tests/fixtures/mini_bdd_raw/det_dirty.json` — one frame each for `labels: null`, an
  unknown category (`trailer`), a degenerate box, a fully-out-of-frame box, an edge-straddling box,
  and a label with no `box2d`.

### Why

Per the plan §5: this is what makes spec 01 testable offline at all. Acceptance criteria 2, 4, and
5 all require fixture inputs that did not previously exist (a raw-format fixture, a dirty fixture,
and all-10-class coverage).

### How

Assigned classes to scenes explicitly (`pedestrian/rider/car`, `truck/bus/train`,
`motorcycle/bicycle`, `traffic light/traffic sign`) instead of extending the random-sample call,
specifically so coverage is guaranteed rather than probable. Kept the box-drawing logic (position,
colour, watermark) otherwise unchanged, and extended `PALETTE` to all 10 classes.

**Rejected alternative:** widening the existing `rng.sample(..., k=rng.randint(2,4))` call to
sample from all 10 classes. Rejected because with only 4 scenes there is no `rng` seed guaranteed
to cover all 10 classes, and "probably covers all 10" is exactly the kind of fixture that passes
today and flakes the day someone changes the seed or scene count.

### Issues & resolutions

None. Determinism was verified directly (md5sum before/after a second `make fixtures` run, byte-
identical) rather than assumed.

### Verification

```
uv run python scripts/make_fixtures.py   → wrote 4 images + 10 boxes; wrote raw fixtures
md5sum (annotations.json, det_val.json, det_dirty.json) before vs. after a second run
                                          → identical (DETERMINISTIC)
uv run pytest tests/test_fixtures.py -v  → 8 passed (spec 00's fixture tests, unchanged, still green
                                             on the extended fixture)
uv run pytest                            → 53 passed in 3.11s
uv run ruff check .                      → All checks passed!
```

**Next:** task 4 — `bdd100k.py` converter + drop accounting + `tests/test_bdd_convert.py`, using
the raw and dirty fixtures just committed.

---

## [SEQ-0010] Task 4 — `bdd100k.py` converter + drop accounting

**Date:** 2026-07-15 · **Spec:** 01-data-bdd100k · **Status:** done

### What

- `src/gdp/data/bdd100k.py` — `convert_bdd_to_coco(frames, classes=..., image_width=...,
  image_height=...)` converts raw BDD `det_20` frames (`{"name", "labels": [{"category", "box2d"}]}`)
  into the project's COCO-style dict. Per box: unknown category / no `box2d` / degenerate
  (zero-or-negative area) / entirely out-of-frame are each **dropped and counted by reason**
  (`ConversionStats.dropped`); edge-straddling boxes are **clipped and counted separately**
  (`stats.clipped`), never dropped. Frames with zero surviving boxes **keep their image entry**
  (`images_without_boxes` counted, not silently shrinking the split — H8). `image_id` is the index
  into `sorted(names)`; `annotation_id` is a running counter — same bytes in, same bytes out.
  Also added `verify_image_size()`, a sample-based check against the assumed uniform BDD100K image
  size, for real-data use (not exercised on the fixture, which is 640×360 by design).
- `tests/test_bdd_convert.py` — 9 tests: `load_dataset` reads the converter's output unmodified
  (acceptance 1); converting the independently-generated raw fixture reproduces the committed
  `annotations.json` (acceptance 2 — a real parity test between two separately-written code
  paths); zero-box frames keep their image entry; deterministic image ids; and the dirty fixture's
  exact per-reason drop counts plus the one clipped-not-dropped survivor (acceptance 5).

### Why

This is where the token-span mapping (task 2) and the drop-accounting design (plan §4) meet real
input for the first time. Building it on the raw/dirty fixtures from task 3 (rather than
hand-written dicts) means the test proves the converter and the fixture generator — two
independently-written pieces of code — agree, which is what makes acceptance criterion 2 ("fixture
and real BDD pass the *same* validation") a real assertion rather than a tautology.

### How

Geometry checks run in a fixed order per box, each guarding one failure mode: unknown category and
missing `box2d` are checked before any arithmetic touches the coordinates; degenerate (`x2<=x1` or
`y2<=y1`) is checked before out-of-frame, since a degenerate box could otherwise be misclassified as
merely out-of-frame; clipping happens last, only after a box has already survived both drop checks.

**Rejected alternative:** clipping before checking degenerate/out-of-frame. Rejected because a box
that is entirely outside the frame would clip to a zero-area box and then need a *second* degenerate
check post-clip — the current order means each box is classified exactly once, by exactly one rule.

### Issues & resolutions

One: the first draft of `verify_image_size`'s type hint used `__import__("pathlib").Path` inline to
avoid a top-level import, which the formatter left in place but which reads as a hack. Fixed by
moving `Path`, `Image`, and `random` to normal top-of-file imports — cosmetic, caught on
self-review before running tests, not by a failing test.

### Verification

```
uv run pytest tests/test_bdd_convert.py -v   → 9 passed in 0.80s (all green on first run)
uv run pytest                                → 62 passed in 3.64s (53 prior + 9 new)
uv run ruff check .                          → All checks passed!
```

The parity test (`test_converted_json_reproduces_committed_annotations`) is the one that matters
most: it compares `convert_bdd_to_coco(raw_fixture)` against the hand-written `annotations.json`
committed in task 3, and both were produced by independent code (`make_fixtures.py`'s COCO writer
vs. `bdd100k.py`'s converter) — so agreement is real evidence, not a tautology.

**Next:** task 5 — `stats.py`, the sanity report (`runs/01-data/<ts>/stats.json`).

---

## [SEQ-0011] Task 5 — `stats.py`, the sanity report

**Date:** 2026-07-15 · **Spec:** 01-data-bdd100k · **Status:** done

### What

- `src/gdp/data/stats.py` — `build_stats(coco, conv_stats, *, split, source_file,
  min_boxes_for_eval=100)` assembles per-class box counts, image counts, `dropped`/`clipped` from
  a `ConversionStats`, `rare_classes` (classes below `min_boxes_for_eval`), `source` (file name +
  sha256), `git_sha`, and a UTC `created` timestamp. `write_stats(stats, spec="01-data")` writes it
  to `runs/01-data/<ts>/stats.json` via the existing `gdp.paths.run_dir` — no new path-handling
  code.
- `tests/test_stats.py` — 5 tests: all 10 classes non-zero on the fixture (acceptance 4);
  `rare_classes` correctly flags every class when the threshold is raised above the fixture's
  per-class count of 1; drop/clip counts pass through from `ConversionStats` unchanged; an explicit
  H9 guard asserting the report contains no key that reads as a metric (`map`, `accuracy`,
  `precision`, `recall`, `score`); and a round-trip through `write_stats` confirming the file lands
  under `runs/01-data/` and is valid JSON with the expected fields.

### Why

Per plan §6/§9: `stats.json` is dataset statistics, never a metric (H9), and real-data class
imbalance (BDD is car-dominated, `train` may be near-zero) must be *flagged*, not smoothed (H7).
Building `rare_classes` here — rather than leaving it to whoever writes the results table later —
is what makes that flag load-bearing rather than aspirational prose in a spec.

### How

Reused `gdp.paths.run_dir` unchanged rather than writing a second path-timestamping scheme.
`git_sha()` shells out to `git rev-parse HEAD` and falls back to `"unknown"` on failure (e.g. no
git binary) rather than raising — a missing git SHA in provenance is a degraded report, not a
reason to crash a working data-prep run.

**Rejected alternative:** computing `rare_classes` against a hardcoded absolute count instead of
`min_boxes_for_eval`. Rejected because the threshold needs to be tunable per-dataset (the plan
ties it to `DatasetConfig.min_boxes_for_eval`, wired in task 6) — a hardcoded number would silently
stop being the config's threshold.

### Issues & resolutions

None. The one thing worth noting: the test-added `runs/01-data/` directory is gitignored but its
now-empty parent (`runs/01-data/`) is left behind after the test's cleanup removes only the
timestamped leaf directory — harmless (the directory is untracked and gitignored), not a bug in
the code under test.

### Verification

```
uv run pytest tests/test_stats.py -v   → 5 passed in 0.76s
uv run pytest                          → 67 passed in 3.58s (62 prior + 5 new)
uv run ruff check .                    → All checks passed!
find runs -type f -o -type d           → runs, runs/01-data (empty; gitignored, untracked)
```

**Next:** task 6 — `gdp data prepare` CLI + `DatasetConfig.raw_labels` /
`min_boxes_for_eval` + remove the `data` row from `PENDING` in `tests/test_cli.py`.

---

## [SEQ-0012] Task 6 — `gdp data prepare` CLI + config keys + `PENDING` row removed

**Date:** 2026-07-15 · **Spec:** 01-data-bdd100k · **Status:** done

### What

- `src/gdp/config.py` — `DatasetConfig` gains `raw_labels: str = "data/bdd100k/labels/det_20"`
  (+ `raw_labels_path()`) and `min_boxes_for_eval: int = 100`, with a `validate()` check that it's
  positive. `configs/bdd100k.yaml` gains the matching keys in the same change (unknown keys raise
  by design, so a config declaring one without the other would already fail loudly).
- `src/gdp/cli.py` — `data` is now a Typer sub-app (`data_app`, registered via
  `app.add_typer(data_app, name="data")`) with one command: `gdp data prepare --dataset
  {bdd100k,mini_bdd} --split {train,val,both}`. It resolves the raw input path and images root per
  dataset (`mini_bdd` always reads `tests/fixtures/mini_bdd_raw/det_val.json` against
  `tests/fixtures/mini_bdd`; `bdd100k` reads `<raw_labels>/det_<split>.json` against
  `<root>/<split>`), converts via `convert_bdd_to_coco`, and writes `det_<split>_coco.json` +
  `stats.json` (or `stats_<split>.json` for `--split both`) into one shared `run_dir("01-data")`
  per invocation. Rejects an unknown `--dataset`/`--split`, and rejects any split but `val` for
  `mini_bdd` (the fixture only has a val-equivalent raw file) with exit code 1.
- `src/gdp/data/stats.py` — `write_stats` gained `out_dir` and `filename` parameters (both
  optional, defaulting to the prior behaviour) so the CLI can share one timestamped directory
  across a coco json and its stats report, rather than `write_stats` minting its own timestamp.
- `tests/test_cli.py` — `data` removed from `PENDING` (it now works); the help-surface test
  checks for `"data"` explicitly alongside the still-pending commands; three new tests: `data
  prepare --dataset mini_bdd` writes both files and every class is non-zero in the stats, the
  non-`val` split is rejected for `mini_bdd`, and an unknown `--dataset` is rejected.

### Why

CLAUDE.md §8 requires one config system with no magic constants; `raw_labels` /
`min_boxes_for_eval` had to land in `DatasetConfig` (not a CLI-only flag) so the same values are
visible to `gdp info`, future specs, and the results table, not just to this one command. The
`PENDING` dict in `test_cli.py` is deliberately the ratchet the plan named it: removing the `data`
row is what marks the roadmap as no longer honestly pending on this command.

### How

Reused `run_dir`, `resolve`, `convert_bdd_to_coco`, `build_stats`/`write_stats` — no new
path-handling or conversion logic in the CLI layer, only argument validation and wiring.

**Design call: one `run_dir` per invocation, not one per split.** For `--split both`, both
`det_train_coco.json`/`det_val_coco.json` and `stats_train.json`/`stats_val.json` land in the
*same* timestamped directory, so a single run of `gdp data prepare` produces one coherent,
timestamped artifact bundle rather than two unrelated ones a user would have to correlate by eye.

**Rejected alternative:** letting `mini_bdd --split train` silently fall through to a
`FileNotFoundError` from a missing `det_train.json`. Rejected in favour of an explicit, named
error (`"the mini_bdd fixture only provides a 'val' split"`) — a raw stack trace on a missing file
would look like a real bug rather than an intentional fixture limitation.

### Issues & resolutions

**The formatter-on-save hook stripped unused imports mid-task, twice.** Adding an import in one
`Edit` call and its usage in a *later* `Edit` call meant the post-edit `ruff --fix` hook ran and
deleted the "unused" import before the usage landed — this happened first in `cli.py` (`Path`,
`DatasetConfig`, `resolve`, `run_dir`, the `bdd100k`/`stats` imports) and then again in
`test_cli.py` (`resolve`, `shutil`). Both times the symptom was a `NameError` at runtime that ruff
itself never flagged (it only checks the file as it exists at each edit boundary, not the
sequence). **Fix:** re-added the import and its usage together in a single edit each time, and
re-ran the affected tests to confirm no import silently vanished again. Worth remembering for
every future task: add an import in the same edit as its first use, not a separate one.

### Verification

```
uv run pytest tests/test_cli.py -v   → 11 passed in 0.68s (8 prior + 3 new)
uv run pytest                        → 69 passed in 3.47s
uv run ruff check .                  → All checks passed!
bash scripts/smoke.sh                → SMOKE OK (device: mps)

uv run gdp data prepare -c configs/default.yaml --dataset mini_bdd
  → val: wrote runs/01-data/<ts>/det_val_coco.json and .../stats.json
  → images root: tests/fixtures/mini_bdd

uv run python -c "from gdp.data import load_dataset; ds = load_dataset(
    'runs/01-data/<ts>/det_val_coco.json', 'tests/fixtures/mini_bdd'); print(len(ds), ...)"
  → 4 10   (4 images, 10 boxes — matches the fixture exactly)

stats.json inspected by hand: per_class has all 10 BDD100K_CLASSES at count 1, dropped all-zero,
clipped 0, rare_classes lists all 10 (min_boxes_for_eval=100 default against a 1-per-class
fixture — expected and correct, not a bug: the fixture was never meant to clear that bar).
```

The scratch run directory from the manual verification was removed afterward (`runs/` is
gitignored and untracked; nothing to commit either way).

**Next:** task 7 — `docs/bdd100k-download.md` (registration steps, expected tree, checksum
manifest).

---

## [SEQ-0013] Task 7 — `docs/bdd100k-download.md`

**Date:** 2026-07-15 · **Spec:** 01-data-bdd100k · **Status:** done

### What

Added `docs/bdd100k-download.md`: registration steps (bdd-data.berkeley.edu, the two archives to
pull), the expected `data/bdd100k/` directory tree matching `configs/bdd100k.yaml`'s
`root`/`raw_labels`, a `sha256sum ... > CHECKSUMS.sha256` manifest step tied to what
`build_stats`'s `source.sha256` field records, the raw `det_20` label format (including the
`labels: null` and no-`box2d` cases the converter handles), the `verify_image_size` usage example
for checking the 1280×720 assumption against a real sample before a full conversion, and the exact
`gdp data prepare --dataset bdd100k --split both` invocation to run once the data is in place.

### Why

Registration may take days (plan §11 / spec §7 risk) — this document is what the user runs once it
completes, and per the plan it should not block spec 01 from being complete and merged on the
fixture in the meantime.

### How

Cross-checked every path and function name in the doc against the code actually written in tasks
4–6 rather than writing from the spec's prose alone: `DatasetConfig.root` /
`raw_labels`/`raw_labels_path()`, `gdp.data.bdd100k.verify_image_size`'s real signature and
defaults, and `build_stats`'s `source` field name. A doc that describes a function signature that
doesn't match the code is worse than no doc.

### Issues & resolutions

None. Doc-only change; no source touched.

### Verification

```
uv run pytest           → 69 passed in 3.73s (unchanged — no code touched)
uv run ruff check .      → All checks passed!
```

Every path and API reference in the new doc (`DatasetConfig.root`, `.raw_labels`,
`verify_image_size(images_root, file_names, ...)`, `build_stats`'s `source` field,
`gdp.config.BDD100K_CLASSES`) was checked against the actual source in `src/gdp/config.py` and
`src/gdp/data/{bdd100k,stats}.py` rather than assumed from the spec.

**Next:** task 8 — final `/verify` pass across the whole spec, `progress_report.md` closing entry,
`specs/README.md` status → `done`.

---

## [SEQ-0014] Spec 01 complete — final verify, status → `done`

**Date:** 2026-07-15 · **Spec:** 01-data-bdd100k · **Status:** done

### What

Ran the full verification command list from `.claude/plans/01-data-bdd100k.md` §10 end to end
against the finished code (not assumed from individual task runs), then set
`specs/01-data-bdd100k.md` and `specs/README.md` to `done`. All 8 tasks from the spec's `## Tasks`
checklist are now checked off.

Acceptance criteria, and what discharges each, verified in this pass:

| # | Criterion | Discharged by |
|---|-----------|---------------|
| 1 | `prepare` output is read by `load_dataset` unmodified | `test_bdd_convert.py::test_converted_json_loads_via_spec00_loader`; also confirmed live via `gdp data prepare --dataset mini_bdd` → `load_dataset(...)` → `4 10` |
| 2 | Fixture and real BDD pass the *same* validation | one converter code path; `test_converted_json_reproduces_committed_annotations` — converting the independently-generated raw fixture reproduces the hand-written `annotations.json` |
| 3 | Each class's span decodes to its class name | `test_prompt.py::test_span_round_trips`, all 10 classes incl. both multi-word ones |
| 4 | `stats.json` written; all 10 counts non-zero | `test_stats.py::test_stats_reports_all_ten_classes`; confirmed live in the CLI-written `stats.json` |
| 5 | Dropped-box count reported explicitly | `test_bdd_convert.py::test_dirty_frames_are_counted_by_reason` — exact counts per reason, plus the one clipped-not-dropped case |
| 6 | All of it exercised on the fixture | every test above runs with zero downloads; `mini_bdd` is the default `--dataset` |

### Why

CLAUDE.md §9 ("before you say done"): `/verify` must pass with real commands and real output, and
`progress_report.md` + `specs/README.md` must be current before a spec is called finished. Running
the *whole* command list fresh (rather than trusting each task's individual verification) is what
catches any interaction between tasks that no single task's test suite would surface.

### How

```
uv sync                                              → resolved, no changes (already synced)
uv run pytest                                        → 69 passed in 3.85s
uv run ruff check .                                  → All checks passed!
uv run python scripts/make_fixtures.py               → wrote 4 images + 10 boxes; wrote raw fixtures
git diff --stat tests/fixtures/                      → non-empty only because this session's
                                                        earlier fixture-extension commit hasn't
                                                        been made yet (working tree vs. last commit,
                                                        which predates spec 01); byte-for-byte
                                                        determinism across repeated runs was
                                                        separately confirmed via md5sum in SEQ-0009
                                                        and holds here too
uv run gdp data prepare -c configs/default.yaml --dataset mini_bdd
                                                      → wrote det_val_coco.json + stats.json
bash scripts/smoke.sh                                → SMOKE OK (device: mps)
```

The scratch `runs/01-data/<ts>/` directory produced by this pass's manual `gdp data prepare`
invocation was deleted afterward (`runs/` is gitignored and untracked, so nothing was ever at risk
of being committed); `git status --short` confirms only intended files are modified/untracked.

### Issues & resolutions

None new. The two real issues from this spec were both caught and fixed earlier: the 4-vs-10-class
fixture contradiction (SEQ-0006, resolved by extending the fixture in task 3) and the
Edit-then-later-Edit import-stripping pattern with the formatter-on-save hook (SEQ-0012, resolved
by adding an import and its first use in the same edit going forward).

### Verification

See the table above — all 6 acceptance criteria mapped to a passing test or a live CLI check.
Final state: **69 tests passing, ruff clean, smoke green**, spec 01's own 8-task checklist fully
checked, `specs/01-data-bdd100k.md` and `specs/README.md` both read `done`.

**What spec 01 hands to spec 02 (zeroshot-baseline):** `gdp.data.prompt.PromptSpans` (the
class↔token-span mapping spec 02's grounding-accuracy evaluation reads spans from), the
`gdp.data.bdd100k.convert_bdd_to_coco` + `verify_image_size` pair for whenever real BDD100K
registration completes, and the `gdp data prepare` CLI as the single entry point for turning either
the fixture or the real dataset into the COCO-style schema spec 00's loader already speaks. Real
BDD100K registration is still the long-lead item (docs/bdd100k-download.md §1) — it blocks running
`prepare --dataset bdd100k` for real, but not spec 02's design work on the fixture.

**Next:** spec 02 (`02-zeroshot-baseline`) — currently `draft`; needs approval before
implementation, per the same process this session followed for spec 01.
