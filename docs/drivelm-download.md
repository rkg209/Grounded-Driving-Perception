# DriveLM + nuScenes download & registration

Neither DriveLM nor nuScenes is redistributable, and neither is ever committed to this repo
(`.gitignore` blocks `data/`). This is the checklist for getting both onto the cluster so
`configs/drivelm.yaml` + `gdp data prepare-drivelm --dataset drivelm` (spec 06) can run against
them. Nothing in `specs/06-data-drivelm.md` requires this document to exist first — every
acceptance criterion is discharged on the synthetic fixture (`--dataset mini_drivelm`). This is the
handoff for when the real data is ready.

**Two separate registrations are required.** DriveLM-nuScenes is built on top of nuScenes camera
imagery, but the images and the QA annotations are hosted, licensed, and downloaded independently.

## 1 · Registration

### 1a. nuScenes (the camera imagery + scene metadata)

1. Register at <https://www.nuscenes.org/> (Motional) — free for non-commercial research use,
   requires accepting the dataset license.
2. From the **nuScenes** download page, get the **`v1.0-trainval`** release:
   - **Image/sensor blobs** — split across several archive parts (`v1.0-trainval01_blobs.tgz`
     through `v1.0-trainval10_blobs.tgz` at last check; the exact part count has changed across
     nuScenes releases, so **verify the current part list on the download page** rather than
     trusting a number here). This project only needs the camera images (`samples/CAM_*/`), not
     LiDAR or radar sweeps (H4: camera-only) — if the download page offers a camera-only subset,
     prefer it; otherwise the full blobs still work, the unused sensor files are simply never
     read.
   - **Metadata blob** — `v1.0-trainval_meta.tgz`. This is small (a few MB) and is the one that
     matters most here: it contains `scene.json`, the token→name join table
     `gdp.data.splits.load_scene_meta` reads (spec 06 design decision 2).
3. This is a large, rate-limited download — do it once, on the cluster, never on the laptop
   (CLAUDE.md §4).

### 1b. DriveLM (the QA annotations)

1. Follow the registration/download instructions at the DriveLM project
   (`OpenDriveLab/DriveLM` on GitHub, and its DriveLM-nuScenes dataset card on Hugging Face) — the
   exact link and any access-request step can change; use whichever the project's own README
   currently points to.
2. Download the **answered train** annotation file — named `v1_1_train_nus.json` at the time this
   doc was written (DriveLM has revised its release naming across versions; confirm the filename
   against the project's current release notes). This is the *only* file with public answers —
   the challenge val/test set is questions-only, answers withheld behind the EvalAI leaderboard
   server (spec 06's status note, `specs/06-data-drivelm.md`). Spec 06 does not need — and this
   pipeline never reads — a DriveLM-provided val/test file; our val comes from partitioning the
   answered train file by nuScenes' own official scene split (§4 below).

## 2 · Expected directory tree

Unpack both so the tree matches what `configs/drivelm.yaml` declares:

```
data/
├── drivelm/
│   └── v1_1_train_nus.json         # DriveLM's answered train file (§1b)
└── nuscenes/
    ├── samples/
    │   ├── CAM_FRONT/               # .jpg, filenames from nuScenes' sample_data
    │   ├── CAM_FRONT_LEFT/
    │   ├── CAM_FRONT_RIGHT/
    │   ├── CAM_BACK/
    │   ├── CAM_BACK_LEFT/
    │   └── CAM_BACK_RIGHT/
    └── v1.0-trainval/
        └── scene.json                # from the metadata blob (§1a)
```

This matches `DriveLMConfig.annotations` (`data/drivelm/v1_1_train_nus.json`),
`DriveLMConfig.nuscenes_root` (`data/nuscenes`), and `DriveLMConfig.scene_meta`
(`data/nuscenes/v1.0-trainval/scene.json`) in `configs/drivelm.yaml`. DriveLM's own
`image_paths` entries are relative to the nuScenes root (`"samples/CAM_FRONT/xxxxx.jpg"`), which is
why `nuscenes_root` and the DriveLM annotation file are configured separately — the converter joins
them (`images_root / rel` in `gdp.data.drivelm.convert_drivelm`) rather than assuming a shared
prefix. `gdp data prepare-drivelm` writes converted output to `runs/06-data/<timestamp>/`, never
back into `data/`, which stays exactly as downloaded.

## 3 · Checksum manifest

Before converting, record a checksum for the annotation file, so "the same split" (H8 — every
later VQA number depends on it) is checkable, not just asserted:

```bash
sha256sum data/drivelm/v1_1_train_nus.json data/nuscenes/v1.0-trainval/scene.json \
  > data/drivelm/CHECKSUMS.sha256
```

`gdp data prepare-drivelm` records the sha256 of the annotation file it converts in the `source`
field of `stats_{train,val}.json` (`gdp.data.drivelm_stats.build_drivelm_stats`), so a later run
can be checked against `CHECKSUMS.sha256` to confirm it used the same input. `CHECKSUMS.sha256`
itself is not committed (it lives under `data/`, gitignored) — keep a copy alongside the dataset on
the cluster.

## 4 · The official split — what this pipeline actually partitions by

**This is not the DriveLM leaderboard split.** DriveLM-nuScenes' challenge val/test answers are
withheld behind EvalAI (§1b). Instead, `gdp data prepare-drivelm` partitions DriveLM's *answered
train* file by the **official nuScenes `v1.0-trainval` train/val scene lists**
(`nuscenes.utils.splits` — 700 train scenes, 150 val scenes), vendored verbatim with provenance in
`src/gdp/data/nuscenes_splits.json`. Every artifact this pipeline writes carries a `caveat` field
naming this explicitly (`gdp.data.drivelm_stats.CAVEAT`) — never silently. See
`specs/06-data-drivelm.md`'s "Status note" section for the full resolution and why.

## 5 · Raw annotation format (DriveLM-nuScenes)

`v1_1_train_nus.json` is a JSON object keyed by 32-hex **nuScenes scene token** (not scene name —
§4's split lists key by name, hence `scene.json`'s token→name join):

```json
{
  "cc8c0bf57f984915a77078b10eb33198": {
    "key_frames": {
      "e3d495d4ac534d54b321f7ed48a04ac5": {
        "image_paths": {
          "CAM_FRONT": "samples/CAM_FRONT/n015-2018-...jpg",
          "CAM_FRONT_LEFT": "samples/CAM_FRONT_LEFT/n015-2018-...jpg",
          "CAM_FRONT_RIGHT": "samples/CAM_FRONT_RIGHT/n015-2018-...jpg",
          "CAM_BACK": "samples/CAM_BACK/n015-2018-...jpg",
          "CAM_BACK_LEFT": "samples/CAM_BACK_LEFT/n015-2018-...jpg",
          "CAM_BACK_RIGHT": "samples/CAM_BACK_RIGHT/n015-2018-...jpg"
        },
        "key_object_infos": { "<c1,CAM_FRONT,1088.3,497.5>": { "...": "..." } },
        "QA": {
          "perception": [{"Q": "What objects...", "A": "There is a car <c1,CAM_FRONT,...>.", "tag": [2]}],
          "prediction": [{"Q": "...", "A": "...", "tag": [0]}],
          "planning": [{"Q": "...", "A": "...", "tag": [1]}],
          "behavior": [{"Q": "...", "A": "...", "tag": [0]}]
        }
      }
    }
  }
}
```

`tests/fixtures/mini_drivelm/v1_1_mini_nus.json` (`scripts/make_fixtures.py`) is a synthetic
4-scene instance of exactly this shape — cross-check the real file's top-level structure against it
before a first real run if anything here looks off.

**Edge cases `gdp.data.drivelm.convert_drivelm` handles, each counted by reason, never silently
dropped (H7):**

- A QA pair with an empty `"Q"` or `"A"` — `missing_qa_text`.
- A category key outside `perception`/`prediction`/`planning`/`behavior` — `unknown_category`.
  DriveLM's own JSON spells it `"behavior"` (US); the charter/spec prose says "behaviour" (UK) —
  the converter copies DriveLM's key verbatim (`gdp.config.DRIVELM_CATEGORIES`).
- A `key_frames` entry missing any of the six `image_paths` camera keys — `missing_image_path`.
- An `image_paths` entry whose file doesn't exist under `nuscenes_root` — `image_file_missing`.
- A QA pair with a missing or empty `"tag"` — `missing_tag`. `tag` is DriveLM's own
  scorer-routing field (a `list[int]`; see `third_party/drivelm/PROVENANCE.md`'s routing table) —
  spec 08's official scorer cannot route an item without one, so it travels through
  `DriveLMRecord.tag` verbatim, byte-for-byte like the QA text, never inferred from `category`
  (confirmed: real DriveLM QA has both `tag=[0]` and `tag=[2]` items *within* `perception`).
- A scene token absent from `scene.json`, or a scene name in neither official split list — a
  **hard error** (`KeyError`), not a drop: silently dropping scenes would shrink the split every
  later number is computed on (spec 06 design decision 2).

Object-reference tags embedded in answers (`<c1,CAM_FRONT,1088.3,497.5>`) are copied byte-verbatim
into `answer` — they are the official ground-truth string spec 08 scores against (H2) — and parsed
separately into the derived, analysis-only `object_tags` field.

## 6 · Running the real conversion

Once the tree and checksums are in place:

```bash
uv run gdp data prepare-drivelm -c configs/default.yaml -c configs/drivelm.yaml \
    --dataset drivelm --split both
```

Add `--train-fraction 0.N` for a scene-level subsample of *train only* (val is never subsampled —
rejected outright if combined with `--split val`). Then read
`runs/06-data/<timestamp>/stats_{train,val}.json` (H9: dataset statistics, not a metric):
`official_split`, `split_provenance`, and `caveat` should be present verbatim on both, `scenes`
should be close to 700 (train) / 150 (val) minus any dropped scenes, and `category_distribution`
should have all four categories non-zero on both splits. Cross-check zero scene-token overlap
directly:

```bash
comm -12 <(jq -r .scene_token runs/06-data/<ts>/train.jsonl | sort -u) \
         <(jq -r .scene_token runs/06-data/<ts>/val.jsonl | sort -u)
# must print nothing
```
