# Grounded Driving Perception — Project Write-up

**Author:** Rahul, IIT Bombay
**Repo:** `4_Honda` (Grounded Driving Perception / "Grounded Driving Intelligence System")
**Timeline:** July 2026, ongoing
**Target:** Honda R&D — Applied Data Scientist (Job 4), DNN Development Engineer (Job 6), AI Research Engineer (Job 7)

This document is my own record of how I built this project, from the first idea to where it
stands today: the reasoning behind every major decision, what actually broke while I was building
it, how I fixed it, and — most important for anyone evaluating this as a portfolio piece — an
honest account of what is measured, what is implemented but not yet measured, and why.

---

## 1. Why I built this

I have a portfolio of nine projects going into my Honda applications, most of them deep on
text-only LLMs (training internals, fine-tuning, agents, retrieval) or backend systems (a broker, a
ledger, a matching engine). None of them touch computer vision, object detection, or
vision-language models — and a Honda R&D employee I spoke to was specific about what their team
actually looks for: "object detection, grounding, VLM-related projects." That is a gap no amount of
LLM-internals depth fills, and it happens to sit exactly on top of what Honda's own job postings
ask for:

- **Job 6 (DNN Development Engineer):** "image recognition algorithms and machine learning models
  for object recognition", "edge device deployment of machine learning models."
- **Job 7 (AI Research Engineer):** *required* — "experience training and deploying vision-based AI
  models"; *preferred* — "research experience using LLMs with RAG, VLM, or model fine-tuning."
- **Job 4 (Applied Data Scientist):** lists "VLM/LLM" directly in its tools line.

So the brief I set myself was: build a driving-perception system that (a) finds objects in a scene
from a natural-language description rather than a fixed class list, and (b) answers questions about
a driving scene the way a vision-language model would, and prove both of those things with real,
citable numbers rather than a demo GIF. It also had to extend rather than duplicate what I'd already
proven — I had already fine-tuned a text LLM with QLoRA in an earlier project, so doing the same
adaptation move on a *vision-language* model here reads as a progression, not a repeat.

## 2. How it actually started

The project began as a single markdown file at the repository root:
`grounded-driving-perception-spec.md`. Before writing a line of code I wrote out, in plain prose,
what the system was, why it mattered to these specific job descriptions, what it explicitly would
*not* do, the architecture in ASCII, the datasets, the eight-weekend build plan, and — the part I
kept coming back to — a list of the exact sentences I wanted to be able to say to an interviewer
without flinching. That charter file is still in the repo, unedited, as the source of truth for
what "done" means.

I made that file effectively immutable early on. Any correction to it later gets appended
elsewhere (a corrections section in the README) rather than silently rewritten — I wanted a record
of what I originally believed versus what I learned, not a document that quietly reads as if I got
everything right the first time.

## 3. The compute constraint that shaped every decision after this

My development machine is an Apple M4 laptop with 16 GB of unified memory. That is enough to write
code, run tests, debug data pipelines, export and quantize a model, and benchmark inference latency
— **it is not enough to fine-tune Grounding-DINO or a 3-billion-parameter vision-language model.**
Not slowly, not at reduced batch size — just not at all; the memory pressure alone stalls the
machine.

I have access to the IIT Bombay SLURM cluster for anything that actually needs a GPU with real
memory. So before writing spec 1 I decided on a hard rule that everything downstream had to respect:

| Runs on my laptop | Runs on the cluster |
|---|---|
| tests, linting, CLI, config | detector fine-tuning |
| synthetic-fixture pipelines | VLM LoRA fine-tuning |
| ONNX export + INT8 quantization | full-split evaluation on real data |
| **latency benchmarking** (the M4 genuinely is a legitimate edge-class target) | anything touching more than a few hundred real images |

I never let myself kick off a real training run interactively in a terminal session on the laptop —
training jobs get written as SLURM scripts and handed to the cluster's scheduler, never launched by
hand in the moment. I actually wrote a small guard script into my own workflow that refuses to run
anything that looks like a training launch, a bulk delete, or a `git add` on the raw dataset
directories, specifically so a moment of carelessness at 1 a.m. can't undo a day of work or leak a
non-redistributable dataset into git history.

The second consequence of the laptop constraint was bigger: **since I don't have the real
datasets downloaded yet** (BDD100K and nuScenes/DriveLM both require registration and are tens of
gigabytes), nothing would be testable at all unless I built a synthetic stand-in. So early on I
wrote `scripts/make_fixtures.py`, which draws a handful of crude driving scenes — sky, road
surface, a lane line, coloured rectangles standing in for a car, a bus, a pedestrian, a traffic
light — and writes them out in COCO-style annotation format with ground truth that is exact
*because I drew it myself*. Every image is watermarked "NOT REAL DATA" in the corner, and there's a
test that asserts the watermark and the dataset description string are both present, specifically
so a fixture number can never be mistaken for a real result down the line. The entire codebase —
CLI, config system, data pipeline, and later the ONNX export path — runs end to end on my laptop
with zero downloads and zero GPU, against this synthetic mini-BDD100K and a matching synthetic
mini-DriveLM I built the same way for the VQA side.

## 4. The rule I held myself to hardest: no unearned claims

The single biggest risk with a portfolio project like this isn't that it's incomplete — incomplete
is normal and explainable. The risk is that it *overclaims*, and an interviewer asks one follow-up
question that shows the whole thing is soft. So before writing any implementation code I wrote
myself a short set of rules and kept them pinned at the top of my working notes for the whole
project:

1. **No pretraining from scratch.** I adapt pretrained backbones and measure the domain gain —
   that's the actual industry workflow, and I say so rather than apologizing for not training a
   detector from random weights.
2. **The VQA stage is scored only on DriveLM's official split, with its official metric.** Never a
   question I invented myself, never a benchmark I graded myself.
3. **"Edge-deployed" means quantized + exported to ONNX + latency-benchmarked on edge-class
   compute.** It never means "running in a car." I catch myself every time that phrase gets close to
   implying the second thing.
4. **Camera only.** No LiDAR, no radar, no sensor fusion — I don't have the sensors or the data, and
   claiming otherwise would be a straightforward lie.
5. **No tracker, no trajectory predictor, no planner as a system module.** I do let the
   vision-language model *answer* DriveLM's official planning/behaviour questions in natural
   language, because that's literally what the benchmark asks — but that's answering a question, not
   running a planning module, and I keep that distinction explicit everywhere it matters.
6. **Detection and VQA are two separate models, evaluated separately, always.** I never let a
   sentence imply one joint system. Any place I compose their outputs (the demo) labels every line
   with which model actually produced it.
7. **Failures get shown, not hidden.** Anything the system can do outside what I've actually
   measured is a demo, and I label it a demo.
8. **No metric without its paired baseline on the same split.** A fine-tuned mAP number is
   meaningless to me without the zero-shot number next to it; a fine-tuned VQA accuracy is
   meaningless without the base-model accuracy next to it, on the same data.
9. **No benchmark I invented myself, no metric I invented myself** — with one narrow, fully
   documented exception (see §7 below on the grounding phrase set), because if a capability has no
   official ground truth, it doesn't get an accuracy number attached to it. It gets called a demo.

I still run through this checklist by hand before any number goes into the README. It has caught
real problems — more on that below.

## 5. Planning the work: spec by spec, not "build it all at once"

I decided early that I would not just start writing detector code. I broke the whole project into
eleven numbered specs and wrote each one out before touching implementation:

| # | Spec | What it produces |
|---|------|-------------------|
| 00 | Scaffold | package skeleton, config system, CLI shell, test harness, synthetic fixtures |
| 01 | BDD100K data pipeline | BDD100K → COCO-style conversion, class-to-prompt-span mapping |
| 02 | Zero-shot baseline | the mAP number the fine-tune has to beat |
| 03 | Detector fine-tuning | zero-shot → fine-tuned mAP delta (Stage-1 headline result) |
| 04 | Grounding evaluation set | a curated set of descriptive phrases + grounding accuracy |
| 05 | ONNX deployment | INT8 quantization, ONNX export, accuracy-vs-latency curve |
| 06 | DriveLM data pipeline | official nuScenes-scene-based train/val split for DriveLM |
| 07 | VLM fine-tuning | LoRA fine-tuning of Qwen2.5-VL-3B on DriveLM |
| 08 | VQA evaluation | base vs fine-tuned accuracy on DriveLM's official metric, per category, failure analysis |
| 09 | Integrated demo | one interface: type a query → see boxes; ask a question → get an answer |
| 10 | Write-up | generated metrics tables, architecture diagram, reproduction instructions |
| 11 | Temporal demo (stretch) | multi-frame reasoning, explicitly a demo with no metric — first thing to cut if time runs out |

Each spec has to state, in a section I call its "honesty contract," which of the nine rules above
it's at risk of violating and how, before I let myself write any code against it. Each spec also has
falsifiable acceptance criteria — "works well" doesn't count; "mAP is written to a metrics file
alongside the zero-shot baseline on the same split" does.

For every spec I followed the same loop: turn the spec into a short implementation plan, break the
plan into small independently-testable tasks, implement exactly one task, verify it with real
commands and real output (never "looks right"), and write down what happened in a running project
journal before moving to the next task. I never batched multiple tasks into one sitting's commit —
partly discipline, partly because it makes it much easier to find where something broke later.

I drew one hard line into the plan from day one, which I called the checkpoint: **by the end of
spec 05, the project is already complete and defensible on its own** — "a fine-tuned, grounded,
edge-deployed driving detector, with measured gains and a published accuracy-vs-latency curve."
Specs 06 through 08 (the VQA stage) are additive on top of that, not a prerequisite for it. That
meant if my schedule slipped, the fallback wasn't "ship something half-broken," it was "ship the
detector story and treat the VQA story as a stretch." Cut order, if it ever came to that: spec 11
first, then spec 09's extras, then specs 06–08 as one block. Specs 00–05 were never on the table to
cut.

## 6. Architecture

Two stages, deliberately kept as two separate models sharing a theme rather than a joint system:

```
STAGE 1 — GROUNDED DETECTION
  camera frame + text query
        │
        ▼
  Grounding-DINO (IDEA-Research/grounding-dino-tiny), fine-tuned on BDD100K
        │
        ▼
  boxes + confidence scores per query
        │
        ▼
  evaluated by: mAP (class-name queries, zero-shot vs fine-tuned)
              + grounding accuracy (descriptive-phrase queries, IoU ≥ 0.5)
        │
        ▼
  quantize (INT8) → export to ONNX → latency benchmark → accuracy-vs-latency curve

STAGE 2 — DRIVING-SCENE QA
  camera frame + natural-language question
        │
        ▼
  Qwen2.5-VL-3B, LoRA fine-tuned on DriveLM
        │
        ▼
  natural-language answer
        │
        ▼
  evaluated by: DriveLM's official metric, base model vs fine-tuned, per category,
              with an ungrounded-answer diagnostic and sampled failure cases

PRESENTATION LAYER — scene report (demo only, no accuracy number of its own)
  detector boxes + VLM answers composed into one readable report,
  every line labelled with which model produced it
```

I picked Grounding-DINO for Stage 1 because it's open-vocabulary out of the box and — this mattered
a lot once I actually checked — the `transformers` implementation accepts real labels and computes
a proper bipartite-matching loss now, so I didn't need to pull in MMDetection just to fine-tune it.
I checked this against the library documentation before committing to the model, because if it had
turned out to be inference-only, the entire Stage-1 plan would have needed to change. I also
checked, before committing, whether it could even be exported to ONNX at all — its attention kernels
are custom CUDA/C++ and don't export by default, but there's a config flag
(`disable_custom_kernels=True`) that swaps in a pure-PyTorch attention path specifically so the
model *can* be exported. I built that flag into my config from the very first spec rather than
discovering the problem at the deployment stage.

I picked Qwen2.5-VL-3B for Stage 2 because it's small enough to fine-tune with LoRA inside a
cluster job's memory budget, ships a real image processor, and is genuinely current — a 2-8B
compact VLM is the right size class for an "edge-adjacent" vision-language story rather than
reaching for something that only makes sense on a datacenter GPU.

DriveLM is the VQA benchmark. It's nuScenes-based, has official train/val scene lists, and — this
is the detail that made it the right choice for the "risk reasoning" part of the pitch — its
official question categories *include* planning, behaviour, and prediction questions. That meant I
didn't need to invent a separate risk-assessment module with my own rules and my own scoring; I
could measure risk reasoning as accuracy on DriveLM's own planning/behaviour category, against real
ground truth, with the model's score sitting right next to the base model's score on the same
split. I considered building a hand-rolled hazard-scoring layer on top of the detector — "distance
to nearest object in the lane," things like that — and rejected it outright: BDD100K is monocular
with no calibration or depth ground truth, so "distance in metres" isn't something I can actually
compute, and grading my own rules against my own labels would be exactly the kind of self-invented
benchmark rule 9 above forbids. If a heuristic risk overlay shows up anywhere in the demo, it's
badged as an unmeasured illustration, full stop, with no score attached to it anywhere.

## 7. The one deliberate exception: a benchmark I built myself

Grounding accuracy — "does the model correctly localize *the cyclist in the right lane* rather than
just *a cyclist*" — has no existing public benchmark for BDD100K. Rather than skip that evaluation
or fake it, I built a small curated set of 150–300 descriptive phrases myself, over real BDD100K
frames, hand-authored against a sampler tool I wrote that overlays candidate frames for me to look
at and pick from. This is the one place in the whole project where I score against ground truth I
invented myself, and I treat it accordingly: it's documented — construction method, known biases,
and the drop rate of phrases I rejected during authoring, all written down in
`data/grounding_eval/construction.md` — and it's labelled "self-built" everywhere the number
appears, never presented next to the DriveLM or BDD100K numbers as if it carries the same weight.

## 8. Building it, stage by stage

### Scaffold and data pipeline (specs 00–01)

I started with the parts that have nothing to do with modelling: a YAML-to-dataclass config system
that deep-merges configs left to right and rejects unknown keys (so a typo in a config file fails
loudly instead of silently getting ignored), a repo-root-relative path resolver, a seeding utility
that fixes determinism across Python/NumPy/PyTorch and picks the right device in priority order
(CUDA, then Apple's MPS, then CPU), a COCO-style dataset loader, and a CLI shell where only the
`info` command actually did anything yet — every other subcommand existed and named which spec would
fill it in, rather than pretending to work.

I hit a real bug here that's worth writing down because it's the kind of thing that only shows up
much later if you don't catch it: my config loader checked whether a field's declared type was a
dataclass by reading `Field.type` directly. I had `from __future__ import annotations` at the top of
that file for cleaner type hints, which turns every annotation into a **string** at runtime instead
of a live class reference. So `Field.type` was the literal text `"PathsConfig"`, not the class
`PathsConfig` — meaning my "is this a nested config" check was silently `False` for every nested
section, and the whole `paths:` / `detector:` / `vlm:` block of any config would have been passed
through as a raw dictionary. The failure mode is nasty: the config *loads without error*, and you
only find out six hours into a cluster job when something tries to read `.model_id` off a plain
`dict` and blows up. I fixed it by resolving field types with `typing.get_type_hints()` instead of
reading the raw annotation, and wrote two tests that pin the correct behaviour so it can't silently
regress.

Then the BDD100K conversion pipeline: turning BDD100K's native label format into COCO-style
annotations, plus a mapping from class index to the token span it occupies inside the prompt string
Grounding-DINO expects (it takes a period-separated string of class names as its "query" and needs
to know which token range in that string corresponds to which class, to align its loss). I built a
drop-accounting report alongside the converter — every record that gets skipped (malformed box,
unknown class, whatever) gets counted and the count gets written out, so "how many of the original
labels did we actually keep" is always an answered question, not a guess.

### Zero-shot baseline and the mAP wrapper (spec 02)

This is where I found the bug I'm proudest of catching before it corrupted a real result. I wrote a
COCO-mAP wrapper around `pycocotools` (never a hand-rolled AP calculation — official tooling only,
per rule 9) and a fixed-threshold greedy IoU matcher for reporting operating-point precision/recall
at a chosen confidence threshold. While writing a "perfect predictions should score mAP ≈ 1.0" test,
I got exactly 0.5 instead. I stepped into `pycocotools`' own `accumulate()` function to find out why,
and found the actual mechanism: it treats a match array as boolean, and it uses annotation ID `0` as
its internal "no match" sentinel value. If a ground-truth box's real ID *happens* to be `0` — which
mine did, because my converter assigns IDs starting at zero, a perfectly reasonable and common
convention on its own — a correctly-matched detection reads as unmatched, because `0` and "no match"
are indistinguishable once cast to boolean. That's not a hypothetical edge case; it's the very first
annotation in every split my converter produces. I fixed it by shifting every ground-truth
annotation ID by +1 immediately before handing the data to `pycocotools`, confirmed
`pycocotools` already reassigns detection IDs internally so only the ground-truth side needed
touching, and deliberately kept the original id-0 fixture in my test suite as the regression test
for exactly this bug, rather than quietly changing the fixture to avoid tripping it.

I also built a threshold-sweep tool with an explicit leakage guard that only sweeps over the
*training* split, never validation — the operating threshold I report has to be chosen honestly,
not picked by peeking at the numbers I'm about to report.

### Detector fine-tuning (spec 03)

Training config, the label-alignment path (matching my prompt token spans against the model's own
tokenizer output — with a test that pins this against the actual HuggingFace tokenizer, not a
guess), the trainer itself with parameter groups, a NaN-abort safety check, and a JSONL training
log. I built an "overfit gate" — before a real fine-tune ever gets launched on the cluster, the
training loop is first pointed at a tiny slice of data and has to actually drive the loss toward
zero; if it can't overfit ten images, there's no point spending cluster hours on the real run, and
the gate catches a broken training loop before it wastes a GPU allocation. I also wrote the
comparison tool that computes the zero-shot-to-fine-tuned mAP delta mechanically from two metrics
files — it refuses to run if it can't find a matching zero-shot baseline on the same split, so
there's no code path that can produce a fine-tuned number without its required baseline sitting
right next to it.

Everything through this spec runs and is fully tested on my synthetic fixture. What it cannot do on
my laptop is the actual fine-tuning run — that's a SLURM job script, written and ready, waiting on
real BDD100K being registered and downloaded on the cluster.

### Grounding evaluation set (spec 04)

The phrase schema, a validator, and a freeze mechanism that hashes the phrase set once it's locked
in, so nobody (including me, six weeks later) can quietly edit an evaluation set after seeing how a
model scores on it. A frame sampler and overlay renderer so I can actually look at candidate frames
while authoring phrases by hand. Single-phrase grounding scoring, and the same zero-shot-vs-fine-tuned
comparison pattern as spec 03.

This is also where I ran into the nastiest infrastructure problem of the whole project: writing the
frame sampler, my test suite loaded the real Grounding-DINO model from HuggingFace **five separate
times in one test file**, because I'd used a function-scoped test fixture instead of a module-scoped
one, plus two more uncached loads elsewhere in the suite. On a 16 GB machine, that ran the memory up
to roughly 40 GB of pressure and hung the laptop hard enough that I had to force a restart. That was
a wake-up call, and I turned the fix into a permanent rule rather than a one-off patch: any test that
loads real model weights gets tagged with a marker (`model_heavy`), the default test run excludes
that marker entirely so a plain "run the tests" command can never load a real model, and any fixture
that does load real weights has to be scoped to the whole file, not re-created per test. I wrote that
rule down at the top level of my own project notes so it would survive me forgetting the incident —
and I applied it retroactively to every later spec's real-model tests from that point on.

### ONNX deployment (spec 05)

This is the project's actual checkpoint — the point where, if nothing else ever landed, I'd still
have a complete, defensible story: a benchmark harness, ONNX export with a numerical-parity test
(the exported graph has to produce the same outputs, within tolerance, as the PyTorch model it came
from — the real risk in any export step), a dynamic INT8 quantization pass, re-evaluation of mAP
across all three variants (PyTorch, ONNX-fp32, ONNX-int8) so the accuracy cost of quantization is
measured rather than assumed, and a curve generator plotting accuracy against latency.

One honest limitation I had to accept and state plainly rather than paper over: exporting to ONNX
requires freezing the text prompt at export time, because ONNX graphs don't carry a dynamic language
encoder path the way the PyTorch model does. So the exported artifact only answers the fixed 10
BDD100K classes — the open-vocabulary capability that's the whole point of the PyTorch model stays
in PyTorch. I record this explicitly as a `fixed_prompt: true` field in the exported model's metrics
file rather than letting it read as a silent regression.

### DriveLM data pipeline and VLM fine-tuning (specs 06–07)

Vendoring the official nuScenes scene-list split (700 train / 150 val scenes) with its provenance
recorded, a synthetic four-scene DriveLM fixture built the same way as the BDD100K one, a converter
with the same drop-accounting discipline, a scene-level leakage gate (so no near-duplicate frame
from the same driving sequence can end up in both train and val), a stratified train subsampler
sized to what's actually trainable on a shared cluster queue, and Qwen2.5-VL's chat-formatting path
wired through its real processor.

Mid-implementation I found a real gap: DriveLM's own question records carry a `tag` field that the
official scorer needs to route a question to the right sub-metric (accuracy vs. a language-quality
metric vs. an exact-match metric), and I hadn't been carrying that field through my converter at
all. It also turned out not to be inferable from the question's category the way I'd assumed —
checking against real DriveLM data showed that "perception" category questions carry *two different*
tag values depending on the specific question, not one. I went back into the already-implemented
spec 06 converter and added the field properly, rather than working around its absence downstream.

For fine-tuning, I built the answer-only loss mask (the model should only be trained to predict the
answer tokens, not re-predict the question and image tokens it was given) with three separate
assertion checks guarding against the ways that mask logic silently breaks, the dataset and collator
classes, an anchored-regex LoRA-target selector that verifies the vision tower stays frozen (an
unfrozen vision tower on a 3B model would blow the memory and time budget, and silently training it
by accident is an easy mistake to make), and a training loop with real gradient accumulation. All of
it is validated on my laptop against a tiny, randomly-initialized few-layer version of the real
architecture — real forward pass, real backward pass, real LoRA attachment, just megabytes of
weights instead of gigabytes, specifically so I can prove the training *mechanics* are correct
without needing the real 3B checkpoint or a GPU. The actual fine-tuning run against real weights and
real data is, like spec 03, a SLURM script waiting on cluster access and dataset registration — spec
08 never evaluates anything spec 07 didn't actually produce, so there's no evaluation step hiding
inside the fine-tuning spec itself.

### VQA evaluation and failure analysis (spec 08)

I vendored DriveLM's own official scoring code rather than reimplementing it — pinned and hashed, so
I'm scoring against exactly the evaluation logic the benchmark's authors wrote, not my
interpretation of it. Built resumable prediction generation (so a long generation run can pick back
up if it's interrupted rather than starting over), overall and per-category scoring, a
self-defined hallucination diagnostic that I'm careful to label as self-defined everywhere it's
reported (it's not part of the official metric, it's a supplementary signal I added), the
base-vs-fine-tuned comparison gate, and a deterministic, category-stratified sampler for pulling out
failure cases to review by hand.

Actually running the vendored scorer against real predictions (not a mock of it) turned up something
I hadn't expected and had to accept rather than work around: two of the official metric's four
weighted sub-components — a "chatgpt" judge component (40% of the composite weight) and a "match"
component that's internally fused with the same judge — require a live, non-deterministic call to a
paid OpenAI API that this project never makes. That means the official composite `final_score` is
structurally uncomputable here and gets reported as `null`, unconditionally, in every metrics file
this spec produces — not a bug, the only honest value, and I say exactly why next to the number
rather than leaving a blank field that looks like an oversight.

### Integrated demo (spec 09)

One interface: type a query, see boxes; ask a question, get an answer; both against the same driving
scene, composed into a single scene report where every line is tagged with which model produced it
— `[detector]`, `[VLM]`, or `[heuristic]` for the unmeasured risk overlay if it's shown at all. I ran
a full pass specifically checking this panel against my own honesty rules before committing it and
caught a real gap: the panel's "Weights" badge, which is supposed to tell the viewer whether the
fine-tuned adapter is actually loaded, was hardcoded to always show the same text regardless of what
was actually passed in — a small bug, but exactly the kind of thing that would have quietly
misrepresented which model produced an answer. Fixed it so the badge reflects the real loaded
weights every time.

Unlike every spec before it, this one has no cluster blocker at all — the demo runs today, completely,
against the synthetic fixtures, with the zero-shot detector and the base VLM, honestly badging every
panel as such. `--checkpoint` and `--adapter` flags are already wired to swap in the real fine-tuned
artifacts the moment they exist, with no code change needed.

### Write-up (spec 10)

I did not want a README with numbers typed by hand, because a hand-typed number is exactly the kind
of thing that goes stale the moment an underlying run changes and nobody notices. So I built a
report pipeline instead: a snapshot step that reads every `runs/*/metrics.json` file that exists,
records whether it came from real data or a synthetic fixture, and writes a single committed
snapshot file; and a render step that regenerates the README's metrics tables from that snapshot,
with a test that fails the build if a generated block in the README doesn't match what the snapshot
says it should be. Hand-editing a number inside a generated block is caught by CI-equivalent local
tests, not caught by hoping I remember to update it.

I also wrote a script that clones the repository's current committed state into a clean temporary
directory and runs the entire install-and-test path against it, because "works on my machine" is not
the same claim as "works from a clean clone." The first time I ran it, it genuinely failed — three
tests failed that had never failed for me locally. The reason: two optional dependencies (the Gradio
UI library and a scorer dependency that isn't even on PyPI) happen to already be installed on my
machine from earlier work, so tests exercising those code paths had never actually been skipped, even
though they were *supposed* to skip gracefully when those optional extras aren't installed. One of
the skip guards was even checking for the wrong error string entirely — it had never once fired,
which is indistinguishable from working right up until someone without my exact environment runs it.
I fixed both, re-ran the clean-clone script, and this time all seven steps — install, test, smoke
test, both fixture data pipelines, and the README-matches-the-committed-snapshot check — passed with
a clean exit code.

## 9. Testing discipline

I did not treat tests as an afterthought bolted on at the end of each spec — they're written
alongside the implementation, task by task, and the running total is something I check on every
change. As of the last full verification pass, the default test suite (the one that runs with a
bare install, with no real model weights downloaded, and finishes in well under a minute) stands at
**507 passing, 2 skipped, 0 failing**, plus a separate, deliberately-not-default suite of tests
tagged `model_heavy` that load real model weights and get run on purpose, not by accident, whenever
I need that specific coverage. Every generated artifact in the repo — the README's metrics tables,
the architecture diagram, the config system, the drop-accounting reports — has at least one test
asserting its actual output, not just that it "ran without an exception."

I also keep a linter (`ruff`) clean on every commit, and a smoke test that exercises the whole import
→ config → device-selection → fixture → CLI path in one command, so I have a fast single check for
"is the basic plumbing still intact" before I go looking for anything more specific.

## 10. Where the project actually stands right now

I want to be completely direct about this, because it's the part that matters most if someone reads
this before an interview.

**What is real, tested, and complete right now:**
- The entire codebase — config system, data pipelines for both BDD100K and DriveLM, the detector
  fine-tuning loop, the VLM LoRA fine-tuning loop, the ONNX export and quantization pipeline, the
  VQA evaluation pipeline (including the vendored official scorer), the integrated Gradio demo, and
  the generated-report pipeline — is implemented, tested end to end against synthetic fixtures, and
  verified to reproduce cleanly from a fresh clone.
- The ONNX export and INT8 quantization steps have actually been run and produce real byte-level
  compression numbers (a full-precision export around 660 MB down to roughly 180 MB after INT8
  quantization) — that part isn't fixture-only, it's a real transformation of the real pretrained
  weights, just not yet re-benchmarked against the real BDD100K validation split for accuracy.
- 507 automated tests pass on a bare install with no dataset downloads and no GPU.

**What is not yet measured, and why:**
- The headline detection number — zero-shot vs. fine-tuned mAP on BDD100K — needs the real dataset
  registered and downloaded on the cluster, and the actual fine-tuning job run there. Both are
  pending on dataset registration, which is outside my control on a timeline.
- The grounding-accuracy number needs 150–300 phrases hand-authored against real frames, which is a
  human curation step I haven't done yet, followed by the same cluster run.
- The Stage-2 VQA number — base vs. fine-tuned accuracy on DriveLM's official split — needs nuScenes
  and DriveLM registered and downloaded (a separate registration process from BDD100K's), and the
  actual LoRA fine-tuning job run on the cluster.
- The accuracy-vs-latency curve for deployment needs the real BDD100K validation split to compute the
  accuracy half of the curve; the latency half can run today but isn't a result without its accuracy
  counterpart sitting next to it.

Every one of these shows up in the README today as a table that explicitly says "Not yet measured,"
names the exact blocker, and is generated from the same snapshot pipeline that will fill it in with
a real number the moment the corresponding cluster run exists — no code changes needed at that point,
just running `report snapshot` and `report render` again and committing the result.

I decided early that this is the right way to present an in-progress project rather than the wrong
one. It would be easy to fabricate a plausible-looking mAP number, or quietly evaluate on the
fixture data and let a reader assume it was real BDD100K, and I specifically built tooling that makes
both of those things hard to do by accident — a fixture-derived number is stamped `synthetic` in the
snapshot and is mechanically prevented from rendering as a headline result. I would rather show up to
an interview with "here is exactly what's measured, here is exactly what isn't, and here is the
one-line command that will fill in the rest the moment I have cluster time and registered datasets"
than with a number I can't stand behind if someone asks a second question about it.

## 11. What's left

1. Register and download BDD100K on the cluster (the long-lead blocker specs 02–05 are all waiting
   on).
2. Run the zero-shot baseline job, then the detector fine-tuning job, on real data.
3. Hand-author the 150–300 grounding phrases against real frames, then run the grounding evaluation
   job.
4. Re-run the deployment benchmark against the real validation split to get the accuracy half of the
   accuracy-vs-latency curve.
5. Register and download nuScenes + DriveLM (a separate registration from BDD100K), run the data
   conversion, then the LoRA fine-tuning job, then the evaluation job.
6. Record the fine-tuned demo video once the real checkpoints exist to swap in.
7. Re-run the report pipeline and commit the filled-in README.

None of the remaining work is a design question at this point — every remaining step is "get access
to a GPU allocation and a registered dataset, then run a script that's already written and already
tested."

## 12. Tech stack

PyTorch, HuggingFace Transformers and PEFT (for LoRA), `pycocotools` for official mAP scoring, ONNX
Runtime for export and INT8 quantization, Gradio for the demo interface, `uv` for Python dependency
management, `pytest` for testing, `ruff` for linting and formatting, and the IIT Bombay SLURM cluster
for anything requiring real GPU memory. Config is YAML validated into Python dataclasses; results are
always written to timestamped JSON files under `runs/`, never only described in prose.

## 13. Lessons I'm taking into the interview

- **Verify library assumptions against documentation before they become load-bearing.** Both the
  "can Grounding-DINO even be fine-tuned in this library" and "can it even be exported to ONNX"
  questions could have invalidated a week of planning each if I'd assumed rather than checked, and
  checking them up front took an afternoon.
- **A bug that "loads without error" is more dangerous than one that crashes.** Both real bugs I
  found and fixed — the dataclass type-resolution issue and the pycocotools annotation-ID collision —
  share that shape: the code runs, produces a plausible-looking wrong answer, and only shows up later
  under scrutiny. I now specifically look for that failure mode when reviewing anything that touches
  config loading or third-party scoring libraries.
- **"Works on my machine" and "works from a clean clone" are different claims**, and the gap between
  them is exactly where undeclared dependencies hide. I wouldn't have found the two broken optional-
  dependency test skips any other way.
- **Memory discipline on a 16 GB machine is a first-class engineering constraint, not an
  afterthought.** One hung laptop from an uncached model load taught me to treat every real-model test
  as something that needs explicit scoping and an explicit opt-in marker, from that point forward, in
  every subsequent spec.
- **Say what isn't measured as clearly as what is.** The instinct when a project is 80% of the way to
  a headline number is to write the README as if it's already there. I think the more defensible
  version — and the one I'd rather be asked about — is a README that names its blockers precisely and
  proves, with a working generation pipeline, that filling them in is a mechanical last step rather
  than more design work.
