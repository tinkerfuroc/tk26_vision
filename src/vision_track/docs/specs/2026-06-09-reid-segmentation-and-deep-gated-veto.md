# ReID precision: deep-feature segmentation + deep-gated color veto + transparent crops

**Date:** 2026-06-09
**Scope:** `vision_track` ReID — make the tracker stop locking onto bystanders and
stop over-rejecting correct matches ("yellow is right but never goes green").
**Constraint:** precision is sacred. We do NOT lower the deep/lock thresholds
(0.72 bars, 0.55 reid_threshold, 0.92 deep-ratio, 0.10 distinctiveness,
MIN_REID_SIMILARITY_RAW 0.40). We remove *contamination* and a *blunt veto*; the
strict bars become correct once the embedding is clean.

---

## Root causes (investigated, with bag-log corroboration)

1. **Hard color veto** — `reid/reid.py:736/749/761`: body/upper/lower color
   histogram-intersection `< 0.40` → `return 0.0` **before** any weighting,
   discarding the 0.75-weighted deep term. A 0.9 deep match is force-zeroed when
   a color score dips to 0.39 (bag logged `lower color similarity 0.388 < 0.4`).
   This is the top cause of correct (yellow) matches not promoting to green under
   lighting/pose/background change.
2. **Background-contaminated deep feature** — `reid/reid.py:253` (single) and
   `:310-316` (batch): the deep OSNet embedding is extracted from the **raw bbox
   crop**, not the mask. The gallery bakes in background + any co-bbox bystander,
   so (a) the tracker can lock onto another person in the box, and (b) the
   same-person cosine drops when the background changes → forces the strict bars
   to over-reject. The gallery stores only this (unmasked) deep vector
   (`core/reid_gallery.py`), and live query + gallery both go through
   `extract_features`/`extract_features_batch`, so a fix there is automatically
   symmetric.

## Decisions (operator)
- Color veto → **gate on deep confidence** (a confident deep match bypasses the
  color floor).
- Deep crop → **dilate + tight-crop + soft background** (OSNet takes 3-ch RGB, so
  a literal transparent/RGBA crop is not an option for the model; this is the
  equivalent).
- Transparent crops → **gallery thumbnails AND on-disk vision_log crops**.

---

## Change 1 — segment the deep OSNet crop (functional)

Add a shared helper in `reid/reid.py`:

```python
def _segment_crop_for_reid(self, crop, mask_crop):
    """Person-segment a crop for the deep embedding: dilate the mask, tight-crop
    to it, soft-attenuate the background. OSNet takes RGB (no alpha), so this is
    how we feed it a 'transparent-background' person. mask_crop None -> passthrough
    (no seg model)."""
```
- Dilate `mask_crop` ~6 px (3×3 kernel, 2 iters) so we keep the silhouette and
  tolerate loose YOLO-seg edges.
- Tight-crop `crop` (and the dilated mask) to the mask's bbox — this is what
  evicts a bystander sitting in the corner of a loose person box and re-centers
  the person for the 128×256 resize.
- For pixels inside the tight crop but outside the dilated mask: blend toward a
  Gaussian-blurred copy of the crop (soft attenuation), NOT hard zero — keeps the
  input in OSNet's training distribution while removing the bystander's identity.
- `mask_crop is None` → return `crop` unchanged (preserves no-seg-model behavior).

Apply at BOTH deep-forward inputs, identically (the `test_reid_batch.py`
row-equivalence test is the symmetry guard):
- Single: `reid.py:253` → `extract_features(self._segment_crop_for_reid(crop, mask_crop))`.
- Batch: `reid.py:310-316` → re-slice `masks[i]` to the bbox and append
  `self._segment_crop_for_reid(frame[y1:y2,x1:x2].copy(), mask_crop_i)` to the
  batch crops (currently the mask is dropped before the batch forward).

No threshold changes. (Re-tuning MIN_REID_SIMILARITY_RAW / REID_THRESHOLD for the
cleaner operating point is arena-deferred.)

## Change 2 — gate the color veto on deep confidence (functional)

In `_compute_person_similarity` (`reid/reid.py`), `reid_sim_raw` (the gallery
deep cosine) is already computed (`:714`). Add:
```python
DEEP_CONFIDENT_BYPASS = 0.70   # arena-tunable; clearly above bystander deep (~0.47-0.57)
deep_confident = reid_sim_raw is not None and reid_sim_raw >= DEEP_CONFIDENT_BYPASS
```
Then change each color veto (`:736`, `:749`, `:761`) from
`if X < floor: return 0.0` to
`if X < floor and not deep_confident: return 0.0` (log "color low … but deep
confident … not vetoing" on the bypass). Keep the raw-deep floor (`:718`)
unchanged.

Why this preserves precision: a bystander has LOW deep (bag showed rejected
candidates at reid 0.45-0.47) → `deep_confident` False → color veto still fires →
still rejected. Only a high-deep correct match whose color drifted bypasses. The
mask-clean deep (Change 1) further widens the gap.

## Change 3a — transparent gallery thumbnails (dashboard)

The gallery `thumb` is currently an opaque RGB crop encoded JPEG
(`_maybe_publish_gallery`). Make it person-only:
- Build the thumb as RGBA with the mask as alpha (background fully transparent)
  where the thumb is created (`reid/appearance_manager.py` admission path).
- Encode PNG (JPEG has no alpha) in `person_track_node._maybe_publish_gallery`.
- Render as a PNG data URL in `webui/app.js` gallery rendering.

## Change 3b — transparent on-disk vision_log crops

When gallery crops / vision logging are enabled, also write the segmented person
view to the vision_log dir as an **RGBA PNG** (mask alpha), so the saved crops
are person-only too. Reuse the same RGBA build as 3a.

---

## Tests
- Existing reid tests stay green — `test_reid_batch.py` (batch==loop
  row-equivalence: the segmentation helper MUST be applied identically in both
  paths) and `test_reid_fp16.py`.
- New `test_deep_crop_segmentation.py`: helper tight-crops to the mask bbox and
  attenuates outside-mask pixels; `mask_crop=None` → identity passthrough.
- New `test_color_veto_deep_bypass.py`: low color + high deep (`>= bypass`) → not
  vetoed (non-zero); low color + low deep → vetoed (0.0); thresholds unchanged.
- Full suite green; flake8 baseline 534 unchanged.
- Bag/live verify: previously-rejected correct matches promote; low-deep
  bystanders still rejected (no new false locks).

## Build/deploy
`tkbuild tk26_vision --packages-select vision_track` → canonical
`/home/tinker/tk25_ws/install`.
