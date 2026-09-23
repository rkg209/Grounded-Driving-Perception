# PROVENANCE — vendored DriveLM challenge scorer

**DO NOT EDIT the files in this directory.** Not to fix a crash, not to add a feature, not to
silence a warning. If the vendored code doesn't do what we need, the fix belongs in
`src/gdp/vqa/official.py` (the adapter) or in a documented note here — never here. Editing these
files would make our headline VQA number the product of code nobody outside this repo has reviewed,
which is exactly what H2/H9 (CLAUDE.md §2) forbid.

## What

| File | Upstream path |
|---|---|
| `evaluation.py` | `challenge/evaluation.py` |
| `gpt_eval.py` | `challenge/gpt_eval.py` (imported by `evaluation.py` for the ChatGPT sub-metric) |
| `extract_data.py` | `challenge/extract_data.py`: selects and **tags** the QA the challenge evaluates (added 2026-09-24, [SEQ-0144]) |

## Where from

- **Repo:** https://github.com/OpenDriveLab/DriveLM
- **Commit SHA:** `266b570f1c746a4cad7f8f0317eb3c2f99141512`
- **Commit date:** 2024-07-12
- **Retrieved:** 2026-07-26
- **License:** Apache License 2.0 (repository-level, per GitHub's license detection for
  `OpenDriveLab/DriveLM`)

Fetched verbatim via:
```
curl -s https://raw.githubusercontent.com/OpenDriveLab/DriveLM/266b570f1c746a4cad7f8f0317eb3c2f99141512/challenge/evaluation.py
curl -s https://raw.githubusercontent.com/OpenDriveLab/DriveLM/266b570f1c746a4cad7f8f0317eb3c2f99141512/challenge/gpt_eval.py
curl -s https://raw.githubusercontent.com/OpenDriveLab/DriveLM/266b570f1c746a4cad7f8f0317eb3c2f99141512/challenge/extract_data.py
```
`extract_data.py` was retrieved 2026-09-24 at the same pinned commit; sha256
`4665d3e101347bcc21061dad248d03a4af2d390dc7ed81e2b596d434ac9d0507`, byte-identical to `main` on that
date.

## Integrity

`sha256(evaluation.py)` is stamped into every `metrics.json` this repo writes (see
`gdp.vqa.official.scorer_sha256`), so any later divergence from the pinned commit is visible in the
artifact itself, not just in this file.

## Where the `tag`s come from (`extract_data.py`)

DriveLM's answered train file (`v1_1_train_nus.json`) carries **no** `tag` on any of its 377,956
QA items; progress_report [SEQ-0143] found this when a converter that required one kept 0 of them.
The tags `evaluation.py` routes on are assigned by `extract_data.py`, which also *selects* the few
QA per key frame that the challenge evaluates. Rules, read from the pinned file:

| Category | Selected (per frame) | Tag |
|---|---|---|
| perception | first QA whose answer contains every object class (`Visual_description.split('.')[0]`) | `[2]` |
| perception | first "What is the moving status of object" question | `[0]` |
| prediction | first QA whose answer contains every object location key | `[3]` |
| prediction | first QA whose answer contains the substring "yes" or "no" | `[0]` |
| planning | first of each: "What actions could the ego vehicle take" / "lead to a collision" / "safe actions" | `[1]` |
| behavior | all | `[0]` |

Rules are kept exactly as written, including the substring yes/no test (which also matches
"not", "know", …) and re-tagging when one QA matches two rules. Fixing them would be inventing a
benchmark (H9). `gdp.data.drivelm_official.official_eval_tags` runs the file unmodified (its
`print()`s are captured, not removed) and maps its output back onto our records by
(scene, frame, category, question, answer). `gdp.data.drivelm`: every **val** record is an
extract_data selection; **train** keeps every clean QA, with `tag: []` where unselected, because tags
only route scoring. The file's sha256 is stamped into `stats_{train,val}.json` → `tag_source`.

**Consequence, given the known gap below:** planning is tagged only `[1]` (ChatGPT judge), so
**planning receives no official score** in this repo. Prediction's location item is `[3]`
(match, also judge-fused), so prediction is scored through its yes/no item only. Planning answers
are generated and shown qualitatively and labelled unscored. They never get a substitute metric.

## Sub-metric routing and final-score weighting (read out of the vendored file, not from memory)

`evaluation_suit.forward(tag, answer, GT)` routes each QA item by its `tag` field (a list of ints)
into up to four buckets — an item can land in more than one:

| `tag` value | Bucket | Scorer |
|---|---|---|
| `0` | `accuracy` | exact-match `answer == GT`, over items so tagged (multiple-choice style) |
| `1` | `chatgpt` | `GPTEvaluation.forward` — a paid OpenAI (`gpt-3.5-turbo`) judge, scores 0–100 |
| `2` | `language` | `language_evaluation.CocoEvaluator(coco_types=["BLEU","ROUGE_L","CIDEr"])`, corpus-level |
| `3` | `match` | `eval_match` = average of (a) coordinate-matching F1 (`match_result`, IoU-free nearest-point matching, fixed distance-16 threshold) and (b) `eval_chatGPT` on the same items — **`match` is itself fused with the paid ChatGPT judge**, confirmed by actually running `evaluation_suit.evaluation()` (see below); it is not separable from `chatgpt` by only skipping the `chatgpt` bucket |

`evaluation.py`'s `__main__` block (lines 194–223 of the pinned file) computes the official
**final score** as a weighted sum, normalized to 0–1 per component:

```
weights = [0.4 (chatgpt), 0.2 (language), 0.2 (match), 0.2 (accuracy)]

chatgpt_component = chatgpt_score / 100
language_component = sum(
    Bleu_1..4 / 4 / 3,   # the four BLEU keys, each divided by 4 then by 3
    ROUGE_L   / 3,
    CIDEr     / 10 / 3,
)
match_component = match_score / 100
accuracy_component = accuracy_score            # already 0-1

final_score = 0.4*chatgpt_component + 0.2*language_component
            + 0.2*match_component   + 0.2*accuracy_component
```

This repo's `gdp.vqa.score` writes this composition verbatim into `metrics.json`'s
`final_score_composition` (design decision 6/7 of the spec-08 plan) and never computes
`final_score` itself when the `chatgpt` component is omitted — see `src/gdp/vqa/official.py` and
`src/gdp/vqa/score.py`.

## Known gap this repo does not paper over

`chatgpt` (weight 0.4) requires a paid, non-deterministic OpenAI API judge. This repo does not run
it (no invented substitute, no rescaled composite — H9).

**Discovered while building the adapter (task 4), not from reading the code alone:** calling
`evaluation_suit.evaluation()` unconditionally calls `eval_chatGPT` — with `language_evaluation`
and `openai` actually installed and zero chatgpt-tagged items forwarded, it raises
`ZeroDivisionError` (`sum([]) / len([])` in `eval_chatGPT`), and with any items it would place a
live paid API call. `eval_match` (the `match` bucket) *also* calls `eval_chatGPT` internally and
averages it with the F1 score — so `match` cannot be computed either without the same paid judge.
`gdp.vqa.official.run_official_scorer` therefore never calls `evaluation_suit.evaluation()`; it
calls only `eval_acc()` (tag 0) and `eval_language()` (tag 2) directly, and reports the
chatgpt/match item counts without scoring them. **Two of the four weighted components — `chatgpt`
(0.4) and `match` (0.2), 60% of the composite's weight — are always in `submetrics.omitted`**, and
`final_score` is written as `null`, never a partial sum. See `metrics.json`'s `submetrics.omitted`
entries and `comparable_to_leaderboard: false`.
