# BDD100K download & registration

BDD100K is not redistributable and is **never committed to this repo** (`.gitignore` blocks
`data/`). This document is the checklist for getting the real dataset onto the cluster so
`configs/bdd100k.yaml` + `gdp data prepare --dataset bdd100k` (spec 01) can run against it. Nothing
in `specs/01-data-bdd100k.md` requires this document to exist before spec 01 ships — every
acceptance criterion is discharged on the synthetic fixture (`--dataset mini_bdd`). This is the
handoff for when the real data is ready.

## 1 · Registration

1. Register at <https://bdd-data.berkeley.edu/> (free, requires an account and accepting the
   dataset license — academic/research use).
2. Download two archives from the "100K Images" and "Labels" sections:
   - **Images:** `bdd100k_images_100k.zip` (~6.9 GB) — contains `train/`, `val/`, `test/`.
   - **Labels:** `bdd100k_labels_release.zip` (the `det_20` detection labels; this project uses
     only `det_20`, not the segmentation or lane labels).
3. Downloads are large and rate-limited — do this once, on the cluster, not on the laptop
   (CLAUDE.md §4: the laptop keeps only the fixture).

## 2 · Expected directory tree

Unpack both archives so the tree matches what `configs/bdd100k.yaml` declares:

```
data/bdd100k/
├── images/
│   └── 100k/
│       ├── train/            # ~70,000 .jpg
│       ├── val/               # ~10,000 .jpg
│       └── test/              # not used by this project
└── labels/
    └── det_20/
        ├── det_train.json     # raw BDD det_20 format (see §4)
        └── det_val.json
```

This matches `DatasetConfig.root` (`data/bdd100k/images/100k`) and `DatasetConfig.raw_labels`
(`data/bdd100k/labels/det_20`) in `configs/bdd100k.yaml`. `gdp data prepare` reads
`<raw_labels>/det_<split>.json` and images from `<root>/<split>/`, and writes converted output to
`runs/01-data/<timestamp>/det_<split>_coco.json` (never back into `data/`, which stays exactly as
downloaded).

## 3 · Checksum manifest

Before converting, record a checksum for each raw label file, so "the same split" (H8 — every
later mAP number depends on it) is checkable, not just asserted:

```bash
sha256sum data/bdd100k/labels/det_20/det_train.json data/bdd100k/labels/det_20/det_val.json \
  > data/bdd100k/labels/det_20/CHECKSUMS.sha256
```

`gdp data prepare` records the sha256 of whichever raw file it converts in the `source` field of
`stats.json` (see `gdp.data.stats.build_stats`), so a later run can be checked against
`CHECKSUMS.sha256` to confirm it used the same input. `CHECKSUMS.sha256` itself is not committed
(it lives under `data/`, which is gitignored) — keep a copy alongside the dataset on the cluster.

## 4 · Raw label format (`det_20`)

Each of `det_train.json` / `det_val.json` is a JSON list of frames:

```json
[
  {
    "name": "0000f77c-6257be58.jpg",
    "labels": [
      {
        "category": "car",
        "box2d": {"x1": 45.2, "y1": 254.2, "x2": 357.9, "y2": 487.5}
      }
    ]
  }
]
```

`labels` may be `null` for a frame with no annotations — `gdp.data.bdd100k.convert_bdd_to_coco`
keeps that frame's image entry with zero boxes rather than dropping it (dropping would silently
shrink the val split). A label may also carry `attributes`-only entries with no `box2d` (dropped,
counted as `no_box2d`) and category names outside the official 10 — real det_20 files ship
ignore-classes such as `other person`, `other vehicle`, and `trailer` beside the 10 official
classes (`gdp.config.BDD100K_CLASSES`). **Verify the exact set of extra category names against the
downloaded file** (`jq -r '.[].labels[]?.category' det_val.json | sort -u`) — the converter counts
and drops any category not in `BDD100K_CLASSES` rather than crashing, but an unexpectedly large
`unknown_category` count in `stats.json` is the signal that the taxonomy assumption above needs
rechecking, not that the dataset is broken.

## 5 · Image size assumption

BDD100K's 100k-split images are documented as uniformly 1280×720. The converter takes this from
`gdp.data.bdd100k.DEFAULT_IMAGE_WIDTH` / `DEFAULT_IMAGE_HEIGHT` rather than reading every image
file (reading 80,000 images just for their header would be needless I/O). Before a full
`gdp data prepare --dataset bdd100k --split both` run, verify the assumption against a sample
instead of trusting it silently:

```python
from gdp.data.bdd100k import verify_image_size
import json

frames = json.loads(open("data/bdd100k/labels/det_20/det_val.json").read())
verify_image_size("data/bdd100k/images/100k/val", [f["name"] for f in frames])
```

This raises immediately, naming the offending file and its actual size, if any sampled image
doesn't match — the intended failure mode is a loud error before conversion, not a silently wrong
`width`/`height` baked into every box's coordinates.

## 6 · Running the real conversion

Once the tree and checksums are in place:

```bash
uv run gdp data prepare -c configs/default.yaml -c configs/bdd100k.yaml --dataset bdd100k --split both
```

Then read `runs/01-data/<timestamp>/stats_val.json` and `stats_train.json` (H9: these are dataset
statistics, not a metric): expect ~10k val images, `car` dominant, and check `rare_classes` — on
real data, `train` may legitimately be near-zero or zero. **If a class other than the ones known to
be rare in BDD100K comes back unexpectedly at zero, the taxonomy mapping in
`gdp.config.BDD100K_CLASSES` is the first thing to recheck, not the dataset.**
