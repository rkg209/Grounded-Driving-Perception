# `tests/fixtures/report/` — synthetic artifacts for the spec-10 renderer

**These numbers are invented. They are not results and must never appear in `README.md`.**

They exist for one reason: today every real stage under `runs/` is either missing (specs 03/04/08,
cluster-blocked) or `is_synthetic: true` (spec 05, run on `tests/fixtures/mini_bdd/`), so the
`present` branch of every table renderer would otherwise ship untested. These files carry
`is_synthetic: false` so that branch is exercised on the laptop today.

The README is rendered from `docs/metrics_snapshot.json`, which is built from `runs/` only — no
code path reads this directory outside `tests/`.

Layout mirrors a real `runs/` tree so `latest_run_dir()` is exercised as-is:

```
runs/03-finetune/20260801-101500/comparison.json
runs/04-grounding/20260801-102000/grounding_comparison.json
runs/05-deploy/20260801-103000/metrics.json + latency.json
runs/08-vqa/20260801-104500/comparison.json + metrics.json
```

Each file deliberately contains a **regression** (a class/qualifier/category where fine-tuning lost
ground) and, for VQA, a **`null` per-category score**, so the tests prove those are rendered rather
than quietly dropped (H7).
