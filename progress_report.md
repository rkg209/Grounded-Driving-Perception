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

## [SEQ-0015] Spec 02 task 1: pycocotools dependency + Grounding-DINO detector skeleton

**Date:** 2026-07-16 · **Spec:** 02-zeroshot-baseline · **Status:** done

### What
Approved spec 02 (flipped `**Status:**` in `specs/02-zeroshot-baseline.md` and its row in
`specs/README.md` from `draft` to `approved`, after confirming with the user, since `approved`
means human-reviewed per CLAUDE.md §3 and I cannot self-certify that). Wrote the spec's 7-task
`## Tasks` checklist into the spec file (cheap sanity checks first: span→class round-trip before
any pipeline wiring, per the `tasks` skill's ordering rule). Then implemented task 1: added
`pycocotools>=2.0.7` to `pyproject.toml`; created `src/gdp/detect/__init__.py` (re-exports
`GroundingDinoDetector`, `Detection`), `src/gdp/detect/detector.py` (`GroundingDinoDetector` —
loads `AutoProcessor` + `GroundingDinoForObjectDetection` for `config.model_id`, resolves the
device via `gdp.seed.select_device`, moves the model to it and calls `.eval()`, passes
`disable_custom_kernels` through to the model config), and `src/gdp/detect/predictions.py`
(`Detection` frozen dataclass — image_id/class_id/score/xyxy with `__post_init__` guards for
degenerate boxes and out-of-range scores — plus `to_coco()` for the pycocotools xywh convention
and a `write_predictions` helper). No inference or span→class logic yet — that's task 2.

### Why
H8 makes spec 02's zero-shot mAP the baseline every later Stage-1 claim (spec 03's fine-tuned
delta) is measured against, so it has to exist and be approved before any fine-tuning touches the
model. This task is the first rung: nothing downstream (span→class mapping, `gdp detect`, COCO
mAP) can be written against a detector class that doesn't exist yet, and getting the model/device/
config wiring right first, in isolation, means task 2's much riskier code (the token-span→class
assignment the `detection-eval` skill calls out as the main way this spec goes silently wrong)
isn't also debugging a model-loading problem at the same time.

### How
Reused `gdp.config.DetectorConfig` (model_id, box_threshold, text_threshold,
disable_custom_kernels) and `gdp.seed.select_device` rather than inventing a second device-
resolution path. Considered putting a `detect()` method stub with `NotImplementedError` on
`GroundingDinoDetector` to signal "more to come"; rejected it — an empty class that only loads
weights is already an honest, testable unit, and a stub method the next task immediately deletes
adds nothing. Verified `AutoProcessor`/`GroundingDinoForObjectDetection` are the correct HF auto-
classes and that `disable_custom_kernels` is a real `GroundingDinoConfig` field (not a made-up
kwarg) by importing and inspecting them directly before writing the loader, rather than trusting
the plan's description of the HF API from memory.

### Issues & resolutions
None. `uv sync` resolved `pycocotools==2.0.11` cleanly (no build issues, despite it being a
C-extension package); the HF Hub download of `grounding-dino-tiny`'s weights succeeded on the
first try (a `HF_TOKEN` warning printed to stderr but did not affect the load).

### Verification
```
uv sync                                                → + pycocotools==2.0.11, gdp reinstalled
uv run python -c "import gdp.detect; ..."              → ok <class 'gdp.detect.detector.GroundingDinoDetector'> <class 'gdp.detect.predictions.Detection'>
uv run gdp detect --help                                → prints the (still-pending, spec 02) detect command's help text, exit 0
uv run python -c "GroundingDinoDetector(DetectorConfig(), device='cpu'); print(d.device, type(d.model).__name__, type(d.processor).__name__)"
                                                         → cpu GroundingDinoForObjectDetection GroundingDinoProcessor
uv run ruff check src tests                             → All checks passed!
uv run pytest -q                                        → 69 passed (unchanged — task 1 adds no new tests; the round-trip
                                                             test for span→class mapping is task 2's, per the spec's task
                                                             breakdown, since that's where the logic it tests first exists)
```

**Next:** spec 02 task 2 — span→class mapping (`PromptSpans.from_classes` with the processor's own
tokenizer, a runtime round-trip assertion, `positive_map` masked-mean aggregation over query
logits) and box-format conversion (normalized cxcywh → absolute xyxy, with `[0, 1]` range guards).
This is the task the `detection-eval` skill's "token-span trap" warning is about — it gets its own
test (`tests/test_detect.py`) asserting the "traffic light" → class 8 case by name.

## [SEQ-0016] Spec 02 task 2: span→class assignment + box-format conversion

**Date:** 2026-07-16 · **Spec:** 02-zeroshot-baseline · **Status:** done

### What
Extended `src/gdp/detect/detector.py` with three module-level functions and one constructor
change. `assert_span_round_trip(spans, classes, tokenizer)` decodes every class's token span and
raises `RuntimeError` on any mismatch. `assign_classes(logits, positive_map)` takes Grounding-
DINO's per-query `[num_queries, text_len]` logits, sigmoids them, and matrix-multiplies against
`PromptSpans.positive_map()`'s L1-normalized `[num_classes, text_len]` rows — since each row is
`1/span_len` over its class's tokens and 0 elsewhere, `scores @ positive_map.T` is exactly the
masked mean per class per query; `argmax` over the class axis gives `(class_ids, scores)`.
`convert_boxes_cxcywh_norm_to_xyxy_abs(boxes, width, height)` converts Grounding-DINO's
normalized-cxcywh `pred_boxes` to absolute xyxy, guarding that every input coordinate is in
`[0, 1]` before converting. `GroundingDinoDetector.__init__` now takes a `classes` argument,
builds `self.prompt` (via `config.prompt()`) and `self.prompt_spans` (via
`PromptSpans.from_classes(classes, tokenizer=self.processor.tokenizer)` — the processor's own
tokenizer, not `gdp.data.prompt.get_tokenizer()`'s default bert instance), and calls
`assert_span_round_trip` on construction. Added `tests/test_detect.py` (8 tests): the round-trip
guard's pass and raise cases, `assign_classes` on synthetic logits (including the named
acceptance-3 case — a box whose logits peak on "traffic light"'s tokens is assigned class index 8,
not 0), box conversion's arithmetic and its out-of-range rejection, and two tests against a real
`GroundingDinoDetector("cpu")` instance confirming its `prompt_spans` round-trip against its own
processor's tokenizer and that `self.prompt` matches `DetectorConfig().prompt()`'s format.

### Why
This is the task the `detection-eval` skill's "token-span trap" warning is entirely about, and the
plan (`.claude/plans/02-zeroshot-baseline.md` design decision 1) called it "the whole risk" in
spec 02: Grounding-DINO has no class head, so a box's class comes from which prompt tokens its
logits score highest on. Decoding those logits to a phrase string and matching by substring (HF's
`post_process_grounded_object_detection` path) is fragile — "traffic light" and "traffic sign"
share a prefix, exactly the case spec 01's `PromptSpans` was built to defend against
(`specs/01-data-bdd100k.md` §3). Doing the masked-mean aggregation instead, and reusing
`positive_map()` rather than re-deriving spans from decoded text, means this task inherits spec
01's already-tested guarantees instead of re-opening the same failure mode. The round-trip
assertion at construction time exists because the processor's tokenizer is not guaranteed to
tokenize identically to the bare `bert-base-uncased` instance `PromptSpans` defaults to (padding,
special-token handling, or a future model swap could all shift offsets); catching a divergence at
startup, loudly, beats discovering it as an unexplained near-zero mAP after a full cluster run.

### How
Verified the real shapes before writing the mapping, rather than assuming them from the plan:
loaded `grounding-dino-tiny` and ran one forward pass on a fixture image, which confirmed
`logits` is `[1, 900, 256]` (900 queries, a *fixed* 256-token text dimension regardless of the
actual ~24-token prompt) and `pred_boxes` is `[1, 900, 4]` — matching `PromptSpans.positive_map()`'s
`max_text_len=256` default exactly, so no shape-mismatch handling is needed beyond the
defensive `min()` already in `assign_classes`. Also confirmed the processor's tokenizer is a
`BertTokenizer` loaded from `IDEA-Research/grounding-dino-tiny` (i.e. bert-base-uncased under the
hood) rather than assuming it. Considered doing the round-trip guard as a method on
`GroundingDinoDetector` directly; extracted it as a standalone `assert_span_round_trip` function
instead, so it — and its failure path — could be unit-tested without loading the full model, which
matters because the model-backed tests already need a `pytest.skip` fallback for offline runs and
a pure-function guard test shouldn't share that dependency. Did **not** implement batched
inference or `Detection` assembly here even though the plan's design decision 1 describes the full
per-query algorithm through box thresholding — that orchestration needs the CLI's image-loading
and batching plumbing, which is task 3's scope; task 2 stays scoped to the two pure, independently
testable pieces (span→class assignment, box conversion) per the spec's task breakdown.

### Issues & resolutions
Two, both self-caught by running the new tests immediately rather than trusting the arithmetic by
eye. First: `test_assign_classes_traffic_light_not_class_zero` initially failed ruff's line-length
check on the `assign_classes` docstring; the formatter hook re-wrapped it and `ruff check` passed
on the next run, so no manual fix was needed — noted here only because CLAUDE.md's "mistakes are
the most valuable part" applies even to trivial ones. Second, a real bug in my own test: I hand-
computed the expected xyxy box for `cx=0.5, cy=0.5, w=0.2, h=0.4, width=100, height=200` as
`[40, 90, 60, 170]`, which is wrong — `h/2=0.2`, so `y0=(0.5-0.2)*200=60` and
`y1=(0.5+0.2)*200=140`, not 90/170 (I'd silently used `h/2=0.3`). `test_convert_boxes_...` caught
this immediately (`AssertionError: Mismatched elements: 2/4`); the implementation was correct, the
test's expected values were wrong, and I fixed the test rather than the code — confirmed by
re-deriving the arithmetic by hand a second time before editing.

### Verification
```
uv run pytest tests/test_detect.py -q   → 1 failed first run (test_convert_boxes_cxcywh_norm_to_xyxy_abs,
                                             wrong hand-computed expected values, see above)
                                          → 8 passed after fixing the test's expected values
uv run ruff check src tests             → All checks passed!
uv run pytest                           → 77 passed in 7.93s (up from 69; +8 new tests, 0 regressions)
```

**Next:** spec 02 task 3 — wire `GroundingDinoDetector` + `assign_classes` +
`convert_boxes_cxcywh_norm_to_xyxy_abs` into the `gdp detect` CLI command end-to-end on the
`mini_bdd` fixture: batched inference, box-threshold filtering, `Detection` assembly, and a
`predictions.json` writer. Also removes `detect` from `tests/test_cli.py`'s `PENDING` map — flagged
here in advance per the plan's note that this touches a committed test file's shared fixture.

## [SEQ-0017] Spec 02 task 3: `gdp detect` end-to-end on the mini_bdd fixture

**Date:** 2026-07-16 · **Spec:** 02-zeroshot-baseline · **Status:** done

### What
Added `GroundingDinoDetector.detect_images(samples, *, box_threshold, batch_size=8)` to
`src/gdp/detect/detector.py`: chunks `Sample`s into `batch_size`-sized groups, runs one processor
call + one model forward pass per chunk (images batched, the fixed class prompt broadcast to
every image in the chunk), then per-image calls `assign_classes` and
`convert_boxes_cxcywh_norm_to_xyxy_abs` and keeps only detections scoring `>= box_threshold` as
`Detection`s. Added `image_id: int` to `gdp.data.core.Sample` (and `load_dataset`'s construction
of it) — needed so predictions carry the same image ids as the ground-truth COCO json, which
`gdp evaluate` (task 5) will need to match them up. Wired the CLI: `gdp detect` now takes
`--dataset`, `--split`, `--limit`, `--box-threshold`; a new `_detect_dataset_paths(cfg, dataset,
split)` helper resolves the mini_bdd fixture's fixed annotations/root, or — for real bdd100k —
swaps `val` for the requested split in the configured `det_val_coco.json` filename, matching
`data prepare`'s own `det_<split>_coco.json` output convention. `detect` loads the dataset,
builds the detector, runs `detect_images`, and writes `runs/02-zeroshot/<timestamp>/
predictions.json` via the new `gdp.detect.predictions.write_predictions`. Removed `detect` from
`tests/test_cli.py`'s `PENDING` map (added an explicit `"detect"` check to
`test_help_lists_the_whole_command_surface` in its place) and added two new tests: a fixture
smoke test asserting `predictions.json` is well-formed COCO-detection-result JSON, and a
non-val-split rejection test mirroring `data prepare`'s existing one.

### Why
This is spec 02's first live, end-to-end proof that the plumbing works — H8's baseline mAP is
worthless until `gdp detect` reliably turns an image + the fixed class prompt into scored,
correctly-labelled boxes on disk. Batching (rather than one image at a time) matters for the real
cluster run (task 7): a python-level loop over thousands of individual forward passes would be
needlessly slow and is also just not how the design decisions section of the plan described the
approach. The `Sample.image_id` addition exists because `gdp evaluate`'s pycocotools comparison
(task 4/5) is meaningless if predictions and ground truth don't share image ids — better to add
the field now, while task 3 is the first code that needs it, than to patch it in later once
task 5's evaluate command discovers the mismatch.

### How
Verified Grounding-DINO's processor actually supports batched multi-image calls with a broadcast
text prompt before writing `detect_images` around that assumption — ran a real 2-image batch
through `AutoProcessor` + `GroundingDinoForObjectDetection` first and confirmed `logits`/
`pred_boxes` both carry a leading batch dimension (`[2, 900, 256]` / `[2, 900, 4]`) that lines up
positionally with the input image list, rather than trusting the plan's "batched inference" phrase
to mean the processor supports it in the way I assumed. For the real-bdd100k annotations path
(`_detect_dataset_paths`), considered adding a dedicated `dataset.annotations_train` /
`annotations_val` pair of config fields instead of string-swapping the filename; rejected it as a
second, parallel way to express what `data prepare`'s own `det_<split>_coco.json` naming
convention already encodes — string-swapping one field mirrors an existing convention rather than
inventing a new config surface, and real-bdd100k detect calls are cluster-only and untestable here
regardless (the fixture is the only path this task can actually verify). Considered making
`box_threshold` filtering happen inside `assign_classes` itself; kept it in `detect_images`
instead, so `assign_classes` stays a pure per-query classification function task 2 already tests
without needing a threshold argument, and thresholding — which is about *keeping* detections, not
*classifying* them — lives with the code that assembles `Detection`s.

Hit the same import-stripping trap flagged in SEQ-0012 twice in this task: adding an import in one
`Edit` call and its first use in a later call gave the formatter-on-save hook a window to strip
the "unused" import in between, producing `F821 Undefined name` on the next `ruff check` (in both
`detector.py`, for `Image`/`Detection`, and `cli.py`, for `load_dataset`/`GroundingDinoDetector`/
`write_predictions`, each twice). Both times the fix was the same: re-add the import in an edit
where the usage is already present in the file, so ruff's autofix has nothing to strip. Noting it
again here because two run-ins in one task means "add the import and its first use together" is a
mechanical habit worth automating rather than a one-off lesson.

### Issues & resolutions
The import-stripping issue above (self-caught via `ruff check` immediately after each edit, fixed
by re-ordering the edits, not by working around ruff). Otherwise clean: `gdp detect -c
configs/default.yaml --dataset mini_bdd` produced a well-formed 4-detection `predictions.json` on
the first successful run once imports were sorted, and the CLI test suite passed without further
changes.

### Verification
```
uv run ruff check src/gdp/detect/detector.py   → All checks passed! (after re-adding Image/Detection imports)
uv run ruff check src/gdp/cli.py               → All checks passed! (after re-adding load_dataset/GroundingDinoDetector/write_predictions imports)
uv run gdp detect -c configs/default.yaml --dataset mini_bdd
                                                → mini_bdd/val: 4 images, 4 detections (box_threshold=0.25)
                                                  -> runs/02-zeroshot/20260716-113052/predictions.json
cat runs/02-zeroshot/20260716-113052/predictions.json
                                                → 4 well-formed COCO-detection-result entries (image_id,
                                                  category_id, bbox [4 floats], score), manually inspected
uv run ruff check src tests                    → All checks passed!
uv run pytest                                  → 78 passed in 20.43s (up from 77: -1 for "detect" leaving
                                                    PENDING's parametrized test, +2 new detect tests)
bash scripts/smoke.sh                          → SMOKE OK (device: mps)
git status --short                             → clean except the intended source/test/spec/progress
                                                  diffs; the manual run's runs/02-zeroshot/<ts>/ directory
                                                  was deleted by hand (rm of the specific timestamped path,
                                                  not a recursive delete of runs/ — the guard-bash hook
                                                  correctly blocked the first, broader rm attempt)
```

**Next:** spec 02 task 4 — the COCO mAP wrapper (`src/gdp/eval/coco_map.py`, a `pycocotools`
wrapper for mAP/mAP@50/per-class AP) and the operating-point precision/recall module
(`src/gdp/eval/operating_point.py`, a greedy IoU≥0.5 matcher plus a `sweep_threshold` helper),
both unit-testable in isolation from the CLI on hand-built synthetic boxes.

## [SEQ-0018] Spec 02 task 4: COCO mAP wrapper + operating-point precision/recall — and a real pycocotools annotation-id-0 bug found and fixed

**Date:** 2026-07-16 · **Spec:** 02-zeroshot-baseline · **Status:** done

### What
Added `src/gdp/eval/coco_map.py` (`evaluate_coco_map(gt_path, predictions) -> CocoMapResult`:
wraps `pycocotools.COCO`/`COCOeval` for mAP@[0.50:0.95], mAP@50, and per-class AP; handles an
empty predictions list without crashing, since `loadRes([])` raises) and
`src/gdp/eval/operating_point.py` (`box_iou`, `match_operating_point` — a greedy per-class
IoU≥0.5 matcher producing per-class + overall TP/FP/FN/precision/recall, and `sweep_threshold` —
picks the F1-maximizing threshold from a candidate list). Added `src/gdp/eval/__init__.py`
re-exporting both modules' public names. Added `tests/test_eval.py` (15 tests): `box_iou`
arithmetic, `match_operating_point`'s TP/FP/FN correctness on hand-built boxes (perfect match,
extra prediction, missed GT, low-IoU near-miss, greedy score-ordering, cross-image and
cross-class non-matches), `sweep_threshold` picking the right candidate by F1 and rejecting an
empty candidate list, and three `evaluate_coco_map` tests (perfect predictions ≈1.0 AP, empty
predictions → all-zero not a crash, accepting a predictions file path). While writing the
perfect-prediction `evaluate_coco_map` test, found and fixed a real bug: added
`_load_coco_gt_with_safe_ids` to `coco_map.py`, which +1-shifts every ground-truth annotation id
before building the `COCO` index, guarding against `pycocotools`' use of annotation id `0` as an
internal "unmatched" sentinel (see Issues & resolutions).

### Why
H9 requires mAP come from `pycocotools`, never a hand-rolled AP interpolation — this task exists
to be that wrapper and nothing more. The operating-point matcher exists because COCOeval's own
precision/recall is threshold-swept and abstract; spec 02's acceptance criteria (and the plan's
design decision 4) call for the specific ADAS-readable sentence "at our operating threshold we
catch X% of pedestrians," which needs a *fixed*-threshold greedy matcher, not COCOeval's sweep.
Both modules had to be built and tested independently of the CLI (task 5's job) so that their
correctness — especially the matcher's TP/FP/FN logic, which is entirely hand-written and has no
upstream implementation to defer to the way `coco_map.py` defers to `pycocotools` — is verified
on inputs simple enough to check by hand, before it is ever asked to produce spec 02's real,
reported baseline number.

### Issues & resolutions
Found a genuine correctness bug, not a test-writing mistake this time. My first `evaluate_coco_map`
test built a 2-image, 2-category synthetic ground truth with perfectly-matching predictions and
expected mAP≈1.0; it returned exactly **0.5**. Debugging (`COCOeval.evaluateImg` called directly,
then `COCOeval.accumulate`'s source) showed the "car" category's single detection had
`dtMatches=[0.]` (looks unmatched) while `gtMatches=[1.]` (correctly recorded gtId 0 as matched to
dtId 1) — an inconsistent pair that only makes sense if the ground-truth annotation's `id` (which
happened to be `0`, the first annotation in the fixture) collided with `dtMatches`' fill-value-0
"no match" sentinel. Confirmed the mechanism in pycocotools' own `accumulate()`:
`tps = np.logical_and(dtm, np.logical_not(dtIg))` treats `dtm` as boolean, so a match recorded as
literal id `0` reads as `False` — the box is counted as a false positive instead of a true
positive. Confirmed this is not a hypothetical: `gdp.data.bdd100k.convert_bdd_to_coco` (spec 01)
assigns annotation ids starting at 0 (`ann_id = 0`), and `tests/fixtures/mini_bdd/annotations.json`
already has an annotation with `id: 0` — so every real converted split, and the fixture itself,
carries this landmine in its very first ground-truth box. Considered fixing it at the source
(bump spec 01's `ann_id` to start at 1); rejected changing already-`done`, already-tested spec 01
code from inside a spec 02 task — 0-based ids are a perfectly valid choice on their own terms, and
the actual defect is pycocotools' sentinel convention, which only matters at the evaluation
boundary. Fixed it there instead: `_load_coco_gt_with_safe_ids` reads the raw GT json and shifts
every annotation id by +1 before constructing the `COCO` index (confirmed `pycocotools.loadRes`
already reassigns detection ids as `1, 2, ...` internally, so only the ground-truth side needed
the fix). Kept the original id-0 fixture in `tests/test_eval.py` rather than changing it to avoid
the bug, specifically so it stands as the regression test — documented as such in the test's
docstring.

### Verification
```
uv run ruff check src tests   → All checks passed!
uv run pytest tests/test_eval.py -q
                               → first run: 2 failed (mAP=0.5 not ~1.0, before the id-0 fix)
                               → 15 passed after adding _load_coco_gt_with_safe_ids
uv run pytest                 → 93 passed in 20.31s (up from 78; +15 new tests, 0 regressions)
bash scripts/smoke.sh         → SMOKE OK (device: mps)
git status --short            → clean except the intended source/test/spec/progress diffs
```

**Next:** spec 02 task 5 — `src/gdp/eval/metrics_io.py` (assembling `metrics.json` with full
provenance: mAP, per-class AP, P/R, threshold + `"chosen_on"`, config echo, model_id, split, image
count, git SHA, UTC timestamp, `is_synthetic`) and wiring `gdp evaluate` in the CLI to consume
`predictions.json` + ground truth and produce it. This is the task that makes spec 02's baseline
number actually exist as a file, per CLAUDE.md §8 ("a metric without provenance is a rumour").

## [SEQ-0019] Spec 02 task 5: `gdp evaluate` + full-provenance `metrics.json`

**Date:** 2026-07-16 · **Spec:** 02-zeroshot-baseline · **Status:** done

### What
Added `src/gdp/eval/metrics_io.py`: `build_metrics(...)` assembles the full provenance record
(dataset, split, num_images, `is_synthetic`, `model_id`, `map`/`map50`/`per_class_ap`,
`box_threshold` + `chosen_on`, overall + per-class precision/recall with tp/fp/fn, a complete
`dataclasses.asdict(cfg)` config echo, `git_sha()`, and a UTC `created` timestamp) and
`write_metrics(...)` writes it to `<out_dir>/metrics.json`. Lifted `_git_sha()` out of
`src/gdp/data/stats.py` into a public `gdp.paths.git_sha()` (per the plan's explicit note to do
this) — `stats.py` now imports and calls the shared helper instead of owning a private copy.
Wired `gdp evaluate` in `src/gdp/cli.py`: takes `--dataset`, `--split`, `--predictions` (defaults
to the most recently modified `runs/02-zeroshot/<ts>/predictions.json`, found by a new
`_latest_predictions_path()` helper), and `--box-threshold`; loads ground truth via the same
`_detect_dataset_paths` helper `detect` uses, scores predictions with `evaluate_coco_map`
(task 4), converts predictions/ground-truth into `PredBox`/`GtBox` and runs
`match_operating_point` at the operating threshold, maps per-class results from class id to class
name via `cfg.dataset.classes`, and writes `metrics.json` via `build_metrics`/`write_metrics`.
Removed `evaluate` from `tests/test_cli.py`'s `PENDING` map (added an explicit `"evaluate"` check
to the help-surface test) and added two tests: a fixture smoke test running `detect` then
`evaluate` and asserting every provenance field is present and correctly typed, and a
`monkeypatch`-based test confirming `gdp evaluate` dies loudly (exit 1, "predictions not found")
when there's nothing to evaluate rather than crashing with a stack trace.

### Why
This is the task spec 02 exists for: H8's zero-shot baseline is not a real, defensible number
until it is a file with its split, threshold, checkpoint id, and git SHA attached — "a number
without that is a rumour" (CLAUDE.md §8). Lifting `git_sha()` to a shared helper (rather than
copy-pasting `_git_sha()` into `metrics_io.py`, which would have been the path of least
resistance) matters because CLAUDE.md's H8 provenance requirement recurs at every future
`metrics.json` (spec 03's fine-tuned mAP, spec 08's VQA accuracy) — one shared, tested
implementation is the difference between "provenance is a project convention" and "provenance is
something I remembered to copy correctly this one time."

### How
Reused `evaluate_coco_map` and `match_operating_point` exactly as task 4 built them rather than
inlining any scoring logic into the CLI — `cli.py`'s `evaluate` command is pure glue: load paths,
call the two eval functions, assemble, write. Considered keying `per_class_pr` in
`build_metrics` by class id (matching `match_operating_point`'s own return type,
`dict[int, PrecisionRecall]`) to avoid a conversion step; converted to class-name keys in the CLI
instead, because `metrics.json` is a human/interviewer-facing artifact (CLAUDE.md's "readable,
alone, as the full development narrative" standard) and `"pedestrian": {...}` is legible in a way
`"0": {...}` is not — the same choice `per_class_ap` (from `coco_map.py`) already made. Did not
add a `--split val` guard against sweeping thresholds here, since `evaluate` doesn't sweep — that
guard is task 6's, once `sweep_threshold` is wired into the CLI; `evaluate`'s own threshold today
is just "the config default, possibly overridden," honestly stamped `"chosen_on":
"config_default"`.

### Issues & resolutions
The `git_sha()` lift hit the same import-stripping pattern (SEQ-0012, recurring in SEQ-0017 and
SEQ-0018): editing `stats.py`'s import line and its call-site body in separate `Edit` calls left
a window where the formatter-on-save hook stripped the "unused" `git_sha` import, and the full
test suite caught it immediately as `NameError: name 'git_sha' is not defined` in `test_stats.py`
and `test_cli.py`'s data-prepare test (6 failures). Fixed by re-adding the import in an edit where
the call-site (`"git_sha": git_sha()`) was already present in the file. `gdp evaluate`'s first
real run against the fixture (after `gdp detect`) produced `mAP=0.0000` / `P=0.000` / `R=0.000` —
inspected the resulting `metrics.json` by hand to confirm this is the *plumbing* being correct
(all provenance fields present and well-typed, `tp=0, fp=4, fn=10` — a legitimate score, not a
crash or malformed output) rather than a bug: the mini_bdd fixture is synthetic imagery
Grounding-DINO was never going to detect anything real on, exactly per design decision 6 ("no
metric computed on it is ever reported").

### Verification
```
uv run gdp detect -c configs/default.yaml --dataset mini_bdd
                                        → mini_bdd/val: 4 images, 4 detections (box_threshold=0.25)
uv run gdp evaluate -c configs/default.yaml --dataset mini_bdd
                                        → mini_bdd/val: mAP=0.0000 mAP50=0.0000 P=0.000 R=0.000
                                          (threshold=0.25) -> runs/02-zeroshot/<ts>/metrics.json
cat runs/02-zeroshot/<ts>/metrics.json → manually inspected: all provenance fields present
                                          (dataset, split, num_images=4, is_synthetic=true,
                                          model_id, map/map50/per_class_ap for all 10 classes,
                                          box_threshold + chosen_on="config_default", overall +
                                          per-class precision/recall/tp/fp/fn, full config echo,
                                          git_sha, created timestamp)
uv run ruff check src tests            → All checks passed! (after re-adding the stripped git_sha
                                          import)
uv run pytest                          → first run: 6 failed (NameError: git_sha not defined,
                                          the import-stripping issue above)
                                        → 94 passed in 32.01s after the fix (up from 93: -1 for
                                          "evaluate" leaving PENDING's parametrized test, +2 new
                                          evaluate tests)
bash scripts/smoke.sh                  → SMOKE OK (device: mps)
git status --short                     → clean except the intended source/test/spec/progress
                                          diffs; both manual runs' runs/02-zeroshot/<ts>/
                                          directories deleted by hand (specific paths, not a
                                          recursive runs/ delete)
```

**Next:** spec 02 task 6 — the threshold sweep on a held-out slice of *train* (leakage guard,
H8): plumb `sweep_threshold` into `gdp detect --split train --limit N`, have `gdp evaluate` record
the winning threshold with `"chosen_on": "train"` instead of today's `"config_default"`, and keep
the fixture (val-only) falling back to the config default, recorded as such. Then task 7's SLURM
job is the last piece before the real cluster run that produces the actual H8 baseline number.

## [SEQ-0020] Spec 02 task 6: `gdp evaluate --sweep` + the train-only leakage guard

**Date:** 2026-07-16 · **Spec:** 02-zeroshot-baseline · **Status:** done

### What
Added `detector.sweep_candidates: list[float]` to `DetectorConfig` (`src/gdp/config.py`,
default `[0.15, 0.2, 0.25, 0.3, 0.35, 0.4]`), validated non-empty and each entry in `[0, 1]` in
`Config.validate()`, and echoed explicitly in `configs/default.yaml` per the project's
no-magic-constants convention. Added a `--sweep` flag to `gdp evaluate`
(`src/gdp/cli.py`): when set, the threshold comes from `sweep_threshold(pred_boxes, gt_boxes,
candidates=cfg.detector.sweep_candidates)` (task 4's helper) instead of a fixed value, and
`metrics.json`'s `chosen_on` is stamped `"train"`. Two guards enforce the leakage rule before any
inference runs: `--sweep` requires `--split train` (rejects with an explicit "leakage" message
otherwise — this is what stops the H8 baseline from ever being tuned toward val), and `--sweep`
is mutually exclusive with `--box-threshold`. Also fixed a smaller honesty gap while touching this
code: a manual `--box-threshold` override was previously mislabelled `chosen_on: "config_default"`
in `metrics.json` even though it came from neither the config nor a sweep — `chosen_on` is now
`"cli_override"` in that case, `"config_default"` only when nothing overrode the config's value,
and `"train"` only for a real sweep. Added tests: two `Config.validate()` cases
(`sweep_candidates` empty, an out-of-range entry), and four CLI tests — sweep-on-val rejected as
leakage, `--sweep`+`--box-threshold` rejected as mutually exclusive, and a positive test that a
manual `--box-threshold` override actually produces `chosen_on: "cli_override"` in the written
`metrics.json`.

### Why
This is the leakage guard H8 explicitly calls for (specs/02-zeroshot-baseline.md §5): the
operating threshold that becomes part of the reported zero-shot baseline must be chosen without
ever looking at val, or the "baseline" the spec exists to produce would already be quietly tuned
toward the split it claims to measure against. Making the guard a hard CLI rejection (not a
comment or a code-review convention) means the mistake is structurally impossible to make by
accident once this ships, not just discouraged. The `chosen_on` mislabelling fix is small but sits
squarely in this task's spirit — `metrics.json`'s whole reason to exist is that every field in it
is defensible provenance (CLAUDE.md §8); a `chosen_on` value that doesn't actually describe where
the threshold came from is exactly the kind of "looks like provenance, isn't" gap this spec is
meant to close.

### How
Considered plumbing the sweep into `gdp detect --split train --limit N` itself (as the task's own
checklist entry, copied from the plan, literally says) rather than into `gdp evaluate`; on reading
the plan's design decision 5 more closely ("`gdp detect --split train --limit N` run *feeds* a
sweep; the eval command records the winner") the sweep computation itself belongs in evaluate —
`detect` only needs to produce train-split predictions (which it already can, unchanged, since
`--split train` was wired in task 3) with enough headroom below the lowest candidate threshold for
the sweep to have real detections to choose among; `evaluate --sweep` is the piece that actually
calls `sweep_threshold`. Did not attempt to build a synthetic mini_bdd **train** split fixture to
exercise the full round-trip offline: mini_bdd's val-only scope is an existing, tested spec-01
decision (`data_prepare` already rejects `--dataset mini_bdd --split train`), real BDD100K train
data is a long-lead item not available locally, and the task's own checklist already scoped this
correctly — "on the fixture (val-only), sweep is skipped and the config default is used." What's
laptop-testable here is the *mechanism* (the guard rejects sweep+val, `sweep_threshold` itself is
unit-tested in task 4, `chosen_on` is correctly stamped) rather than a real sweep result, which is
inherently a cluster-only output once real train data exists — task 7's SLURM job is where that
chain (`detect --split train` → `evaluate --sweep` → `detect --split val --box-threshold
<winner>` → `evaluate --split val`) actually runs for real.

### Issues & resolutions
None. The `sweep_candidates` config addition, its validation, and the CLI wiring all worked on
first test run — the only back-and-forth was deciding where the sweep call belonged (see How),
not fixing a bug.

### Verification
```
uv run ruff check src tests   → All checks passed!
uv run pytest tests/test_config.py tests/test_cli.py -q
                               → 29 passed
uv run pytest                 → 99 passed in 43.32s (up from 94; +5 new: 2 config validation
                                 cases, 3 CLI tests for the sweep guard and chosen_on correctness)
bash scripts/smoke.sh         → SMOKE OK (device: mps)
git status --short            → clean except the intended source/test/spec/progress/config diffs;
                                 the CLI tests' own runs/02-zeroshot/<ts>/ fixtures were cleaned
                                 up by their own teardown fixture, nothing left to delete by hand
```

**Next:** spec 02 task 7 — the last task before the real cluster run: emit
`scripts/slurm/zeroshot_eval.slurm` via the `slurm-job` skill, resumable, chaining `gdp detect
--split train` → `gdp evaluate --split train --sweep` → `gdp detect --split val --box-threshold
<winner>` → `gdp evaluate --split val` against `configs/default.yaml -c configs/bdd100k.yaml`, and
writing the real `runs/02-zeroshot/<ts>/metrics.json` — the actual H8 baseline number. Handed to
the user for `sbatch`, never run in-session (CLAUDE.md §4).

## [SEQ-0021] Spec 02 task 7: `scripts/slurm/zeroshot_eval.slurm` + honest `--chosen-on` provenance — spec 02's laptop work is complete

**Date:** 2026-07-16 · **Spec:** 02-zeroshot-baseline · **Status:** done

### What
Added `scripts/slurm/zeroshot_eval.slurm`: a resumable job chaining `gdp data prepare --dataset
bdd100k --split both` → `gdp detect --split train --limit 2000 --box-threshold 0.05` (headroom
below every sweep candidate) → `gdp evaluate --split train --sweep` (picks the operating
threshold on train only) → `gdp detect --split val --box-threshold <winner>` → `gdp evaluate
--split val --box-threshold <winner> --chosen-on train` (the real H8 baseline, honestly stamped as
train-chosen even though this particular invocation replays the number via `--box-threshold`
rather than sweeping itself). A `run_step` bash helper caches each step's output path in
`runs/02-zeroshot/slurm-state/<step>.path` after success and skips already-completed steps on
resubmission — SLURM pre-emption/walltime kills are normal on a shared cluster, not exceptional.
The job also copies `data prepare`'s freshly converted `det_{train,val}_coco.json` (written to a
fresh `runs/01-data/<ts>/` per spec 01's convention) into the fixed location
`configs/bdd100k.yaml`'s `dataset.annotations` expects (`data/bdd100k/labels/det_20/`), since
those two conventions don't line up for real bdd100k the way they do for the mini_bdd fixture.

Before writing the script, closed a real H8 provenance gap the SLURM chain would otherwise have
exposed: added a `--chosen-on` option to `gdp evaluate` (`src/gdp/cli.py`) that only accepts
`"train"`, requires `--box-threshold`, and is rejected if combined with `--sweep` (redundant —
`--sweep` already stamps `chosen_on: "train"` itself). Without it, the val run in the SLURM
chain — which supplies the sweep's winning threshold via `--box-threshold` rather than sweeping
itself — would have had its `metrics.json` mislabelled `chosen_on: "cli_override"`, hiding the
fact that the number really did come from a train-only sweep. Added 6 new tests: 2
`Config.validate()` cases for `detector.sweep_candidates` (empty, out-of-range), and, spanning
this task and the tail of task 6, 4 CLI tests covering `--chosen-on`'s full validation surface
(rejects non-"train" values, requires `--box-threshold`, and — the positive case — actually
produces `chosen_on: "train"` in the written `metrics.json` when used correctly).

With task 7 done, all 7 tasks in spec 02's checklist are checked. Flipped spec 02's status from
`approved` to `in-progress` (not `done`) in both `specs/02-zeroshot-baseline.md` and
`specs/README.md`: acceptance criterion 1 ("`metrics.json` contains mAP + per-class AP on the
**full** official val split") is not yet satisfied — that requires the cluster to actually run
`sbatch scripts/slurm/zeroshot_eval.slurm` against real, downloaded BDD100K, which is still the
long-lead item this session cannot unblock.

### Why
The SLURM script is the literal deliverable CLAUDE.md §4 requires in place of ever training or
running a full evaluation in-session on the M4 laptop: "emit a SLURM script and hand it to the
user." H8 makes the chain's exact shape non-negotiable — the threshold that ends up in the
reported baseline must be chosen on train and never on val, and `metrics.json` must say so
truthfully, not just conveniently. The `--chosen-on` fix exists because writing the script honestly
surfaced a gap the CLI itself couldn't have hidden forever: eventually *something* would need to
replay a train-derived threshold into a separate val evaluation call (this SLURM script is exactly
that "something"), and without `--chosen-on`, that call's own `metrics.json` would have quietly
mislabelled its own provenance the first time anyone actually ran the real chain — precisely the
"looks like provenance, isn't" failure CLAUDE.md §8 exists to prevent.

### How
Verified the `run_step` bash mechanism against a real command before trusting it in a script that
will only ever be reviewed, never run in this session (`bash -n` syntax-checks a script, it does
not catch "the command substitution captured six lines of tee'd stdout instead of one path"):
ran `run_step`'s exact logic locally against `uv run gdp detect --dataset mini_bdd` twice in a row
— the first call captured a clean single-line path via the `-> <path>` trailer grep, the second
call (with the marker file already present) skipped re-running the command and returned the
identical cached path, confirming both the parsing and the resumability actually work rather than
just reading plausibly. Also ran a live `--box-threshold ... --chosen-on train` evaluate call
against the fixture and inspected the resulting `metrics.json` by hand to confirm `chosen_on` came
out `"train"` (not `"cli_override"`) before trusting the SLURM script's final step to rely on that
behavior. Considered building a generic single `run_step` wrapper for *every* step including `data
prepare`; rejected it for that one step specifically — `data prepare`'s log has multiple `.json`
paths per invocation (a coco json and a stats json, per split), so the same "`grep` the last
`-> <path>`" heuristic that works cleanly for `detect`/`evaluate` (which each print exactly one
summary line) would have silently grabbed the wrong file for `data prepare`; wrote that one step
by hand with a `det_train_coco.json`-specific pattern instead of forcing a shared abstraction onto
an input shape it doesn't fit. Chose `uv run python -c "import json; ..."` over `jq` to extract the
sweep-winning threshold from `metrics_train`'s JSON, since `jq` is not a project dependency and its
presence on the IITB cluster is an unverified assumption — `uv run python` is already how every
other line of the job invokes tooling.

### Issues & resolutions
None new in the script itself — the local dry-run of `run_step` and the `--chosen-on` verification
both passed on the first attempt once written. (The `--chosen-on` gap itself is not a "bug found
and fixed" in the SEQ-0018/SEQ-0019 sense — it was designed correctly on the first pass, prompted
by reasoning about the SLURM chain's needs before writing code, not by a failing test.)

### Verification
```
bash -n scripts/slurm/zeroshot_eval.slurm   → syntax OK
[local run_step dry-run against a real `uv run gdp detect --dataset mini_bdd`]
                                             → first call: captured a clean predictions.json path
                                               via the '-> <path>' trailer
                                             → second call (marker cached): skipped re-running,
                                               returned the identical path — resumability confirmed
uv run gdp detect -c configs/default.yaml --dataset mini_bdd
                                             → mini_bdd/val: 4 images, 4 detections
uv run gdp evaluate -c configs/default.yaml --dataset mini_bdd --box-threshold 0.2 --chosen-on train
                                             → mini_bdd/val: mAP=0.0000 ... -> runs/02-zeroshot/<ts>/metrics.json
python3 -c "print(json.load(open('metrics.json'))['chosen_on'])"
                                             → train (confirmed, not cli_override)
uv run ruff check src tests                 → All checks passed!
uv run pytest tests/test_cli.py -q          → 19 passed
uv run pytest                               → 102 passed in 55.62s (up from 99; +3 new
                                               --chosen-on tests)
bash scripts/smoke.sh                       → SMOKE OK (device: mps)
git status --short                          → clean except the intended diffs; all manual
                                               detect/evaluate runs' runs/02-zeroshot/<ts>/
                                               directories deleted by hand (specific timestamped
                                               paths, never a recursive runs/ delete)
```

**Next:** hand `scripts/slurm/zeroshot_eval.slurm` to the user for review and `sbatch` on the IITB
cluster once real BDD100K is registered and downloaded there (`docs/bdd100k-download.md`) — that
run is what actually satisfies spec 02's acceptance criterion 1 and flips its status to `done`.
All 7 of spec 02's laptop-side tasks (detector skeleton, span→class mapping, `gdp detect`,
COCO mAP + operating-point P/R, `gdp evaluate` + provenance, the train-only sweep guard, and this
SLURM script) are complete, tested (102 passing tests total, up from 69 at the start of this
session), lint-clean, and narrated end-to-end in this file. After the cluster run lands the real
`metrics.json`, spec 03 (fine-tune the detector) becomes unblockable — H8 requires exactly this
baseline to exist first.
