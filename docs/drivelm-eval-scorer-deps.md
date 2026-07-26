# Installing the vendored DriveLM scorer's dependencies

Spec 08's headline VQA number is scored by `third_party/drivelm/evaluation.py`, vendored verbatim
from `OpenDriveLab/DriveLM` (see `third_party/drivelm/PROVENANCE.md` for the pinned commit,
license, and exactly how it routes/weights sub-metrics). Neither of its two Python dependencies is
a required install for this repo — `gdp.vqa.official`/`gdp.vqa.score`'s tests skip themselves with
a clear reason when they're missing, and `scorer_sha256()`/`scorer_composition()` (task 1) never
need them at all. Install both before running `gdp vqa score` for real, or before running
`uv run pytest tests/test_vqa_official.py tests/test_vqa_score.py` and wanting the real
vendored-scorer round-trip tests to execute rather than skip.

## 1 · `openai`

A normal, hard optional dependency — declared in `pyproject.toml`'s `drivelm-eval` extra:

```bash
uv sync --extra drivelm-eval
```

This is needed only so the vendored `evaluation_suit.__init__` can *construct* a `GPTEvaluation`
instance (it does so unconditionally). This repo never calls the OpenAI API — `gdp.vqa.official`'s
`run_official_scorer` deliberately calls only `eval_acc()`/`eval_language()`, never
`evaluation_suit.evaluation()`, which is the only path that would place a live call (H2/H9; see
`third_party/drivelm/PROVENANCE.md`'s "Known gap" section for how this was confirmed by actually
running the vendored code, not just reading it).

## 2 · `language_evaluation`

Not published to PyPI — install per its own README:

```bash
uv pip install "git+https://github.com/bckim92/language-evaluation.git"
uv run python -c "import language_evaluation; language_evaluation.download('coco')"
```

The second command downloads the BLEU/ROUGE/CIDEr/METEOR reference assets (~60MB); only
BLEU/ROUGE_L/CIDEr are actually used (`coco_types=["BLEU", "ROUGE_L", "CIDEr"]` in the vendored
file), and none of those three need Java — the upstream package's own setup notes mention Oracle
Java only for METEOR/SPICE, which this repo never invokes.

## 3 · Verifying the install

```bash
uv run pytest tests/test_vqa_official.py tests/test_vqa_score.py -q
```

If both dependencies are importable, `test_run_official_scorer_end_to_end_on_fixture_predictions`
and the other real-scorer round-trip tests run for real instead of skipping — look for
`OfficialScorerUnavailable` in the skip reason if they don't.

## 4 · On the SLURM cluster

`scripts/slurm/vqa_eval.slurm`'s `score` step needs both installed in whatever environment `uv run
gdp vqa score` executes in. If compute nodes have no outbound network access, install both (and run
`language_evaluation.download('coco')`) on a login node or in the image/environment build step
*before* submitting — the job itself does not attempt to install anything.
