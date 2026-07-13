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
