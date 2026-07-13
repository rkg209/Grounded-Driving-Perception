# Honda ADAS Project — Grounded Driving Perception, Deployed
### (Open-vocabulary road-user detection + language grounding + driving-scene VLM QA, edge-benchmarked)

> The single Honda/ADAS project on the Honda + Integrated (2-page) resume, replacing the earlier trajectory-prediction spec. Written in the standard beginner- and model-readable format: objective, why, what, how, architecture, datasets, scope, build steps, metrics, deployment, and interview-safe one-liners.
>
> **Target roles this project is engineered for:** Honda R&D Job 6 (DNN Development Engineer — "image recognition algorithms and machine learning models for object recognition", "edge device deployment of machine learning models"), Job 7 (AI Research Engineer — *required:* "experience in training and deploying vision-based AI models"; *preferred:* "research experience using LLMs with RAG, VLM, or model fine-tuning"), and Job 4 (Applied Data Scientist — tools line: "VLM/LLM"). It also matches, word for word, the guidance received from a current Honda R&D employee: build "object detection, grounding, VLM related projects."
>
> **Honesty guardrails (read first — these keep every claim defensible):**
> - We **adapt and fine-tune pretrained models; we do not pretrain from scratch.** Say this proudly, not apologetically — adapting foundation models to a domain and proving the gain is exactly how industry vision teams work.
> - The VQA stage is evaluated on a **public driving benchmark (DriveLM / nuScenes-QA) with its official splits and metrics** — not on self-invented questions.
> - **Out of scope and stated openly:** LiDAR/radar, multi-object tracking, trajectory prediction, closed-loop driving, and any claim of real-vehicle deployment. "Edge-deployed" means quantized + exported + latency-benchmarked on edge-class compute, not installed in a car.
> - VLM answers can hallucinate; we **measure** failure modes (accuracy per question type, qualitative failure analysis) rather than hide them.

---

## 1 · Objective (one paragraph)
Build a vision-language perception system for driving scenes that can (a) **find any road user or object described in natural language** — "the pedestrian near the crosswalk", "the cyclist in the right lane" — by adapting an open-vocabulary detector to driving data, and (b) **answer natural-language questions about the scene** — "Is it safe to merge left?", "What should the ego vehicle watch for?" — by fine-tuning a compact vision-language model on a public driving-QA benchmark; then (c) **prove deployability** by quantizing, exporting to ONNX, and publishing the accuracy-vs-latency curve. Every stage is measured: detection mAP (zero-shot vs fine-tuned), grounding accuracy, VQA benchmark accuracy (base vs fine-tuned), and edge latency/FPS.

## 2 · What it is (plain language)
Classic driving perception models are **closed-set**: they detect only the fixed classes they were trained on (car, pedestrian, cyclist…) and understand no language. Modern vision-language models changed this: an **open-vocabulary detector** can localize objects from a free-text description, and a **VLM** can reason about an image in natural language. This project brings both to the driving domain: one system where you can *ask for* objects in language and get boxes, and *ask about* the scene and get answers — the perception style AD teams (including Honda's) are actively adopting because real roads always contain things outside a fixed class list.

## 3 · Why we are building it (portfolio logic + JD mapping)
- **It fills the portfolio's largest empty territory.** The other 8 projects cover text-LLMs deeply (internals, fine-tuning, agents, RAG) and systems deeply (broker, ledger, matching engine), but contain **zero object detection, zero grounding, zero vision-language** — the exact three keywords a current Honda R&D employee advised targeting. Every major component of this project is net-new to the portfolio.
- **It satisfies the hard requirement lines.** Job 7 *requires* "training and deploying vision-based AI models" — Stage 1 fine-tuning + the ONNX/edge benchmark is that, verbatim. Job 6's first responsibility is object recognition; its list also includes "Generative AI" and "development using LLMs", which the VQA stage covers. Job 4's tool list names "VLM/LLM".
- **It extends, rather than repeats, existing skills.** The fine-tuning skill built in LLM Engineering (QLoRA on a text model) extends here to vision-language — creating a narrative arc: *built a text LLM from scratch → fine-tuned and edge-deployed it → extended fine-tuning to vision-language on driving data.* Progression, not repetition.

## 4 · The real problem it solves / who uses it
Fixed-class detectors fail on the open world: an overturned scooter, a person in a costume, construction equipment — anything outside the training taxonomy is invisible. Language-grounded perception lets a system (or a developer/analyst) query scenes flexibly, and scene-level QA supports downstream reasoning, data triage ("find risky scenes"), and in-car assistance. Every AD/ADAS organization is investing in exactly this VLM-ification of perception; within Honda's JDs it appears as object recognition (J6), vision-model training/deployment (J7), and VLM tooling (J4). The audience for the portfolio piece: perception and AI teams screening for candidates who can work with modern vision-language stacks, not just tutorial-level YOLO.

## 5 · What exactly it does (be precise — interviewers will probe)
- **Stage 1 input/output:** input = a single camera frame + one or more text queries; output = bounding boxes with confidence per query. Two evaluation modes: (a) **standard detection** — queries are the dataset's class names, scored with mAP against BDD100K's labeled boxes; (b) **grounding/referring** — queries are descriptive phrases with spatial or attribute qualifiers, scored as grounding accuracy (box IoU ≥ 0.5 with the referred object) on a curated evaluation set.
- **Stage 2 input/output:** input = camera frame(s) + a natural-language question; output = a natural-language (or multiple-choice) answer, scored with the benchmark's official metric on DriveLM / nuScenes-QA held-out splits — covering perception questions ("what objects are ahead?"), and the benchmark's reasoning categories.
- **What it does NOT output:** no tracks (single-frame), no future trajectories, no driving commands. It is the perception + scene-understanding layer; planning/control are downstream and out of scope.
- **Classes:** whatever language describes — that is the point of open-vocabulary. Evaluated rigorously on BDD100K's 10 driving classes + the curated grounding set; capability demonstrated qualitatively beyond them (with the honest caveat that unmeasured classes are demos, not results).

## 6 · Existing alternatives & how ours differs
- **Closed-set detectors (YOLO/DETR fine-tunes):** the standard student project; no language, fixed taxonomy, saturated as a portfolio signal.
- **Zero-shot open-vocab models out of the box:** capable but measurably weaker on driving-specific classes/viewpoints — this gap is precisely what our fine-tuning quantifies and closes (the zero-shot score *is* our baseline).
- **Generic VLM demos ("I asked GPT-4V about a road photo"):** no fine-tuning, no benchmark, no metrics — unfalsifiable.
Ours differs by being **domain-adapted with measured gains** (zero-shot → fine-tuned mAP delta), **benchmark-evaluated** (official DriveLM/nuScenes-QA splits), and **deployment-proven** (quantized, ONNX-exported, latency-curved) — the three things the tutorials skip.

## 7 · Architecture & data flow
```
STAGE 1 — GROUNDED DETECTION
 Camera frame ──► Open-vocabulary detector (Grounding-DINO / OWL-ViT class,
      +           fine-tuned on BDD100K driving classes)
 Text query  ──►  → boxes + confidences per query
                       │
                       ▼
              Evaluation: mAP (class-name queries, zero-shot vs fine-tuned)
                          + grounding accuracy (descriptive-phrase set)
                       │
                       ▼
              Quantize → ONNX export → edge latency benchmark (accuracy-vs-FPS curve)

STAGE 2 — DRIVING-SCENE QA
 Camera frame(s) ──► Compact VLM (Qwen2-VL / LLaVA class, 2–8B,
        +            LoRA fine-tuned on DriveLM / nuScenes-QA train split)
 Question       ──►  → natural-language / multiple-choice answer
                       │
                       ▼
              Evaluation: official benchmark metric, base VLM vs fine-tuned
              + per-category accuracy + failure analysis (hallucination cases)

INTEGRATED DEMO
 One interface: type a query → see boxes (Stage 1); ask a question → see answer (Stage 2),
 side by side on the same driving scenes.
```
The two stages share the theme (language ↔ driving vision) but are **separate models evaluated separately** — stated openly; no claim of one joint model.

## 8 · Datasets (all public, all labeled — no collection needed)
- **BDD100K** — 100K driving images, 10 detection classes, diverse weather/time-of-day; the fine-tuning + mAP evaluation base for Stage 1.
- **Grounding evaluation set** — a curated subset of descriptive queries over BDD100K/nuScenes frames (spatial/attribute phrases), built once, documented, used for grounding accuracy. (Small, honest, and you own its construction — a good interview talking point about evaluation design.)
- **DriveLM (nuScenes-based) or nuScenes-QA** — public driving-VQA benchmarks with official train/val splits and metrics; the fine-tuning + evaluation base for Stage 2. Use a tractable subset for training on the college cluster; evaluate on the official split.

## 9 · Scope boundary (state explicitly — what this project does NOT do)
- **No pretraining from scratch.** We adapt pretrained open-vocabulary and VLM backbones via fine-tuning (full or LoRA) and prove the domain gain. This is the industry-standard workflow and is stated as such.
- **No LiDAR/radar, no sensor fusion.** Camera-only. (J6's fusion bullet is acknowledged as adjacent future work, not claimed.)
- **No tracking, no trajectory prediction, no planning/control.** Single-frame perception + scene QA; the rest of the stack is downstream.
- **"Edge-deployed" means:** quantized + ONNX-exported + latency/FPS measured on edge-class compute (laptop CPU and/or Jetson-class budget if available). It does not mean installed in a vehicle.
- **VQA limits:** the VLM can hallucinate; accuracy is reported per question category with failure cases shown, not hidden.

## 10 · Tech stack
PyTorch; a pretrained open-vocabulary detector (Grounding-DINO or OWL-ViT family) + its fine-tuning recipe; a compact VLM (Qwen2-VL-2B / LLaVA-class) with LoRA/QLoRA fine-tuning (transformers + PEFT — directly reusing the LLM Engineering skill set); BDD100K + nuScenes/DriveLM devkits; ONNX Runtime (+ INT8 quantization; TensorRT if hardware allows); W&B for training curves; matplotlib/Gradio for the demo. Trained on the college GPU cluster; near-zero API cost.

## 11 · How we build it (step by step, ≈7–8 weekends, with a built-in fallback)
1. **W1 — Baseline:** dataset setup; run the pretrained detector **zero-shot** on BDD100K validation with class-name queries; record zero-shot mAP. This number is the bar the fine-tune must beat.
2. **W2–3 — Fine-tune Stage 1:** fine-tune the detector on BDD100K train; re-evaluate; report the zero-shot→fine-tuned mAP delta (the Stage-1 headline). Build the curated grounding-query set; measure grounding accuracy.
3. **W4 — Deploy Stage 1:** quantize, export to ONNX, benchmark latency/FPS vs accuracy; produce the accuracy-vs-latency curve. ◄ **CHECKPOINT: at the end of W4 the project is already complete and defensible as "fine-tuned + deployed driving detector with grounding" — the VQA stage below is additive, so schedule overrun degrades scope, not viability.**
4. **W5–6 — Fine-tune Stage 2:** LoRA fine-tune the compact VLM on the DriveLM/nuScenes-QA train subset; evaluate base vs fine-tuned on the official split; per-category accuracy + failure analysis.
5. **W7 — Integrate + demo:** one Gradio interface — query→boxes and question→answer on the same scenes; record the demo video.
6. **W8 — Write-up:** README with architecture diagram, all metrics tables, failure analysis, honest-scope section, reproduction instructions.

## 12 · Metrics to report (headline + set)
- **Headline (Stage 1):** fine-tuned vs zero-shot **mAP** on BDD100K driving classes — the measured value of domain adaptation.
- **Grounding accuracy:** % of descriptive queries whose predicted box hits the referred object (IoU ≥ 0.5) on the curated set.
- **Headline (Stage 2):** fine-tuned vs base **VQA accuracy** on the official DriveLM/nuScenes-QA split, plus per-category breakdown.
- **Deployment:** latency p50/p99 and FPS after quantization, with the accuracy-vs-latency curve (the J6/J7 "deploy" evidence).
- **Qualitative:** side-by-side examples — zero-shot miss vs fine-tuned hit; VLM correct answers and honest failure cases.

## 13 · What to deploy / show
Architecture diagram + the **metrics tables** (both headline deltas) + the **accuracy-vs-latency curve** + a **demo video** of the integrated Gradio interface (typed query → boxes; typed question → answer). Optionally host the Stage-2 demo on HF Spaces (small model, feasible free); the detector demo runs in the video + notebook. Public repo with reproduction scripts.

## 14 · How Honda uses this (relevance statement, JD-mapped)
Honda's AD/ADAS stack begins with perception — and their hiring documents show the direction: object recognition models (J6), vision-model training and deployment (J7, required), and VLM/LLM tooling (J4). This project implements exactly that slice: a language-grounded, domain-adapted, edge-benchmarked perception layer plus benchmark-evaluated scene understanding — demonstrating, with measured results, the modern vision-language perception workflow their teams are adopting. It deliberately stops at the perception/understanding boundary and states the downstream stack (tracking, prediction, planning) as out of scope — the honest scoping their interviewers respect.

## 15 · Skills it proves
Object detection + evaluation (mAP, IoU); open-vocabulary/grounded detection; vision-language models and VLM fine-tuning (LoRA on a multimodal model); benchmark-based evaluation on public driving datasets (BDD100K, DriveLM/nuScenes-QA); evaluation-set design (the curated grounding set); quantization + ONNX export + edge latency benchmarking; transfer of LLM fine-tuning skills to a new modality; failure analysis and honest scoping; PyTorch + transformers/PEFT + driving-dataset devkits.

---

## Appendix · Interview-safe one-liners
- **"What does it do?"** Two things on driving scenes: find objects described in language (fine-tuned open-vocabulary detection, measured by mAP and grounding accuracy) and answer questions about the scene (a compact VLM fine-tuned and evaluated on the DriveLM/nuScenes-QA benchmark) — both with a quantized, ONNX-exported, latency-benchmarked deployment path.
- **"Did you train the models from scratch?"** No — I adapted pretrained open-vocabulary and VLM backbones and *measured* the domain gain (zero-shot vs fine-tuned). That's the industry workflow; pretraining from scratch would be neither feasible nor sensible here.
- **"Why open-vocabulary instead of YOLO?"** Closed-set detectors can't see outside their taxonomy; real roads always contain out-of-list objects. Open-vocabulary detection localizes from language — and my fine-tuning quantifies how much domain adaptation improves it on driving data.
- **"What's the difference between detection and grounding here?"** Detection scores class-name queries against labeled boxes (mAP); grounding scores *descriptive phrases* ("the cyclist in the right lane") — same model, harder queries, evaluated on a curated set I designed and documented.
- **"Does the VLM hallucinate?"** Yes, sometimes — I report per-category accuracy on the official benchmark split and show failure cases openly rather than hiding them.
- **"Is it deployed in a car?"** No — "deployed" here means quantized, ONNX-exported, and latency-benchmarked on edge-class compute, with the accuracy-vs-latency tradeoff published. Vehicle integration is downstream of this project's scope.
- **"Where does this sit in the AD stack?"** It's the perception + scene-understanding layer. Tracking, prediction, and planning consume its outputs — deliberately out of scope, and I can explain what each downstream layer would need from mine.
- **"Why is this relevant to Honda?"** Job 6 asks for object-recognition models and edge deployment; Job 7 requires training and deploying vision-based models; Job 4 lists VLM/LLM in its tooling — this project is those lines, implemented and measured.
- **"Biggest limitation?"** Camera-only and single-frame — no fusion, no temporal reasoning; and grounding is evaluated on a curated set rather than a large official benchmark. All three are clear, honest next steps.
