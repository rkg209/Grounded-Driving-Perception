# BDD100K download & registration

BDD100K is not redistributable and is **never committed to this repo** (`.gitignore` blocks
`data/`). This document is the checklist for getting the real dataset onto the cluster so
`configs/bdd100k.yaml` + `gdp data prepare --dataset bdd100k` (spec 01) can run against it. Nothing
in `specs/01-data-bdd100k.md` requires this document to exist before spec 01 ships — every
acceptance criterion is discharged on the synthetic fixture (`--dataset mini_bdd`). This is the
handoff for when the real data is ready.

## 1 · Obtaining the data

### 1.1 What we actually used (done 2026-09-22, PARAM Rudra)

**The Berkeley portal <https://bdd-data.berkeley.edu/> returned `403 Forbidden`** (observed
2026-09-16), so the data came from the Kaggle mirror **`awsaf49/bdd100k-dataset`** (v1, uploaded
2020-09-11, 6.38 GB download / 7.4 GB stated size). It was chosen over the other mirror,
`solesensei/solesensei_bdd100k`, after listing every file in both through Kaggle's public API:
awsaf49 has the flat official layout and the 2020 detection labels, solesensei has the 2018 labels
(`person`/`motor`/`bike`) and re-splits train/test into `trainA/trainB/testA/testB`.

```
zip sha256   fe526b971a2c855c239780db7e18cd8b34219643b8b8427d638ca0d58cc77ee8
det_v2_train_release.json  136f76c917cfc6cd6752465cf0d6de47d15f68667f014c23d8a57f279b03f2a2
det_v2_val_release.json    7fd84c4bc2a4d165e3f8c2458f311b83e27dd82a592667684a66522fa853e84b
```

**This is an unofficial mirror, and that fact travels with every number derived from it.** Kaggle
labels it "CC0: Public Domain", which is wrong — BDD100K's own academic/non-commercial license
still applies. What makes it usable as "the official val split" is not the label on the page but
the verification in §1.2: the split sizes and the class histogram match the published dataset.

Download (login node — compute nodes have no internet):

```bash
uv pip install kaggle
kaggle datasets download awsaf49/bdd100k-dataset -p $SCR/4_Honda/data/bdd100k_raw
```

On Rudra the Kaggle token cannot live at `~/.kaggle/kaggle.json` — the home group quota is full and
rejects even a 74-byte write. Put it at `$SCR/.config/kaggle/kaggle.json` (mode 600) and export
`KAGGLE_CONFIG_DIR` to that directory.

The archive also contains a 20k test split, a 10k segmentation set and two CSVs this project never
uses; extracting selectively saves ~1.5 GB and a lot of time:

```bash
unzip -q bdd100k-dataset.zip \
  'bdd100k/bdd100k/images/100k/train/*' 'bdd100k/bdd100k/images/100k/val/*' 'labels/det_v2_*' \
  -d extracted
```

The mirror's label files are named `det_v2_{train,val}_release.json`. Keep those names (they are
the provenance) and add the names the converter expects as symlinks:
`ln -sf det_v2_val_release.json det_val.json`.

### 1.2 The verification that made it trustworthy

Run **before** converting, not after:

| Check | Expected | Observed 2026-09-22 |
|---|---|---|
| val images / label frames | 10,000 | **10,000 / 10,000** |
| train images / label frames | 70,000 images | **70,000 images, 69,863 labeled frames** |
| category names | the 10 in `gdp.config.BDD100K_CLASSES` | **all 10 present**, none missing |
| ignore-classes present | `other vehicle`, `other person`, `trailer` | **85 / 1 / 2** in val |
| labels without `box2d` | some (the 2018 release had poly2d lane/drivable entries) | **0** — this release is detection-only |
| image size | uniformly 1280×720 | **50 sampled per split, all 1280×720** |

The category check is the one that decides whether the converter runs unmodified — it confirms the
mirror ships the 2020 `det_20` taxonomy, not the 2018 one. A `DROP` line naming `person`, `bike` or
`motor` would mean the opposite, and that is a converter change requiring a spec amendment.

**69,863, not 70,000:** 137 train images carry no entry in the detection labels. The converter
builds its COCO from the label file, so the train split is **69,863 labeled frames** and must never
be described as "70k images".

### 1.3 If registering at the portal instead

1. Register at <https://bdd-data.berkeley.edu/> (free, requires an account and accepting the
   dataset license — academic/research use). Returned 403 as of 2026-09-16.
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

On Rudra this ran as `srun --partition=small --cpus-per-task=4 --time=00:30:00 uv run gdp data
prepare …` (job 409040) — it is CPU-only, but 352 MB of train JSON is more than a login node should
carry. **Note:** compute nodes have no `git`, so `git_sha` in the resulting `stats_*.json` is
`"unknown"`. The 2026-09-22 conversion was run from a clean clone at commit `f756beb`; record the
commit by hand when the field is blank.

Then read `runs/01-data/<timestamp>/stats_val.json` and `stats_train.json` (H9: these are dataset
statistics, not a metric): expect ~10k val images, `car` dominant, and check `rare_classes` — on
real data, `train` may legitimately be near-zero or zero.

**Observed 2026-09-22** (`runs/01-data/20260922-225014/`), for comparison on any later re-run:

| | val | train |
|---|---|---|
| images | 10,000 | 69,863 (10 with no boxes) |
| boxes | 185,945 | 1,273,707 |
| dropped (`unknown_category`) | 88 | 1,085 |
| `degenerate` / `out_of_frame` / `no_box2d` / `clipped` | all 0 | all 0 |
| `rare_classes` | `["train"]` (15 boxes) | none (`train` = 128) |

Kept + dropped reconciles exactly against a hand count of the raw label file (185,945 + 88 =
186,033 in val). **If a class other than the ones known to
be rare in BDD100K comes back unexpectedly at zero, the taxonomy mapping in
`gdp.config.BDD100K_CLASSES` is the first thing to recheck, not the dataset.**
