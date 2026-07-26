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
```

## Integrity

`sha256(evaluation.py)` is stamped into every `metrics.json` this repo writes (see
`gdp.vqa.official.scorer_sha256`), so any later divergence from the pinned commit is visible in the
artifact itself, not just in this file.

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
