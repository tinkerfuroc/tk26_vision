# Multi-View ReID Gallery — Design (Spec A)

**Status:** approved design, pre-plan
**Date:** 2026-06-06
**Branch:** `feat/person-tracker-overhaul`
**Related:** [Spec B — Active Re-ID Interface](2026-06-06-active-reid-interface-design.md); benchmark + failure analysis in `benchmarks/person_tracker/ptbench/tpt_bench/DOWNLOAD.md`; memory `vision-track-tracker-gap-analysis`.

## Goal

Raise the person tracker's **reacquisition recall** by comparing reappearance
candidates against a curated bank of *diverse* operator views instead of a single
averaged feature — **without lowering any appearance threshold** and **without
regressing precision**.

## Background & problem

The 2026-06-05 LaSOT `person` benchmark (20 sequences, tracker core) scored mean
**P=0.928 / R=0.490 / F=0.559**. Precision is strong (only 2/20 sequences show a
wrong-lock) — the overhaul's primary goal. The limiter is **recall**: on ~half
the sequences the tracker loses the operator and conservatively refuses to
re-lock (high precision, very low recall).

One root cause is the appearance model used at reacquisition. `TargetAppearance`
keeps `feature_history: Deque[np.ndarray]` (maxlen=30) and reacquisition compares
a candidate against `get_average_feature()` — the **mean** of that deque
(`reid/reid_search.py:44`). During a steady tracking run the last 30 frames are
near-identical (consecutive frames, one pose), so the "history" captures
*recency*, not *diversity*. When the operator reappears in a new pose / scale /
orientation, the candidate scores low against the stale averaged feature and
fails the existing gates — most importantly the **single-candidate guard**
(`reid/reid_search.py:406`, requires similarity ≥ 0.72 when the operator is the
only person visible).

**Constraint (user-set): precision is sacred.** Do not lower `reid_threshold`
(0.55), the distinctiveness margin (0.10), or the single-candidate guard (0.72).
Real-arena distractors are far less similar than the LaSOT identical-costume-crowd
cases, so the crowd-ambiguity (distinctiveness) failures are largely a benchmark
artifact and are **not** a target of this work.

## Approach

Maintain a small **curated gallery of K diverse, high-quality operator views**,
distinct from the rolling `feature_history`, and at reacquisition score a
candidate as the **maximum similarity over the gallery views** rather than
against the mean.

### Why this is precision-safe

- All existing gates are unchanged. The gallery only changes *how the
  candidate's deep-ReID similarity is computed*, then the same
  `reid_threshold` / distinctiveness / single-candidate gates apply.
- Max-over-diverse-views helps the *true* operator (a candidate matches the
  nearest stored pose) far more than a distractor (unlikely to match any one
  specific operator view highly).
- It helps most exactly where it is safest — the **single-candidate** case
  (operator reappears alone but drifted): there is no distractor to false-match,
  so clearing the 0.72 guard via a better-matching stored view is pure recall
  gain. In multi-candidate scenes the distinctiveness gate still guards.

### Guardrail

Validated on the LaSOT `person` proxy with a hard acceptance rule: **mean recall
must rise and mean precision must stay flat (within ±0.01)**, checked
per-sequence as well as in aggregate. If precision drops, fall back to a stricter
scoring mode (configurable): `top2_mean` (mean of the two best gallery views) or
`require_mean` (candidate must also clear the bar against the averaged feature).

## Components

### 1. `ReIDGallery` (new, pure logic)

A bounded, curated collection of operator views. One clear responsibility:
decide what to remember and score a candidate against memory.

- **State:** up to `gallery_size` (K, default 6) entries, each holding the
  L2-normalized deep feature vector (and lightweight metadata: bbox aspect,
  admit-time frame index — for eviction/diagnostics only).
- **Admission (`maybe_add(feature, quality_ok) -> bool`):** add a view only when
  (a) `quality_ok` (the caller passed the existing `reid/quality.py` hygiene:
  min-height, mask-coverage > 0.4, blur, not back-view) **and** (b) it is
  *novel* — its max cosine similarity to existing entries is below
  `gallery_novelty_max` (default 0.85). Novel + high-quality only, so the bank
  spans genuinely different views and never fills with near-duplicates.
- **Eviction:** when full, evict the entry that is *most redundant* (highest mean
  cosine to the others) so diversity is preserved as new views arrive. The very
  first locked view is pinned (never evicted) as the identity anchor.
- **Scoring (`score(candidate_feature) -> float`):** return the max cosine over
  all entries (or the configured fallback mode). Returns 0.0 when empty.
- **No ROS / no torch deps** — operates on numpy feature vectors, unit-testable
  in isolation.

### 2. Integration into the ReID search

- `find_best_match_reid` (`reid/reid_search.py`): replace the
  `target_reid = tracker.target_appearance.get_average_feature()` deep term with
  the gallery's `score(candidate_feature)` for each candidate's deep-ReID
  similarity. Color/body/size terms and the fusion weights are unchanged.
- The gallery is populated during confident tracking: where the tracker today
  appends to `feature_history` under hygiene gating, also call
  `gallery.maybe_add(feature, quality_ok)`. The rolling `feature_history` stays
  (other consumers may use the average); the gallery is the reacquisition memory.
- Reset semantics: `tracker.reset()` clears the gallery (a fresh target). (Spec B
  introduces a gallery-*preserving* re-seed; this spec leaves reset as-is.)

### 3. Configuration (`config/default.yaml`)

- `reid_gallery_enabled` (default `true`) — kill-switch; `false` restores
  `get_average_feature()` behaviour exactly.
- `reid_gallery_size` (K, default 6).
- `reid_gallery_novelty_max` (default 0.85) — admission novelty ceiling.
- `reid_gallery_score_mode` (default `max`; alts `top2_mean`, `require_mean`) —
  the precision fallback knob.

## Data flow

```
tracking frame ─▶ extract deep feature ─▶ quality.py hygiene ─┐
                                                              ├─▶ feature_history (unchanged)
                                                              └─▶ gallery.maybe_add (novel + quality)
operator lost ─▶ reacquire: for each candidate ─▶ gallery.score(cand) = max over views
                                              ─▶ same reid_threshold / distinctiveness / single-candidate gates
```

## Error handling & edge cases

- **Empty gallery** (loss before any view admitted): `score()` returns 0.0 →
  reacquisition fails closed (no spurious match). Identical to "no feature yet".
- **Dimension mismatch** (mixed backbone dims, as `get_average_feature` already
  guards): skip non-matching entries; if none match, fall back to the rolling
  average for that call.
- **Degenerate / all-similar views:** novelty gate simply keeps K small; scoring
  still valid.
- **Kill-switch off:** code path is exactly today's averaged-feature behaviour.

## Testing

- **Unit (pure, no model/dataset):** admission (novelty + quality gating),
  eviction preserves diversity and pins the anchor, bounded size, `score`
  returns max / fallback modes, empty-gallery → 0.0, dim-mismatch handling.
- **Integration:** with a stubbed feature extractor, a drifted reappearance that
  fails against the mean is reacquired against the gallery; a distractor is still
  rejected by the unchanged gates.
- **LaSOT proxy (acceptance gate):** run
  `demo/run_lasot_person_benchmark.py` (all 20 person seqs) with the gallery on
  vs off; **require mean recall↑ and mean precision flat within ±0.01**, and no
  single sequence's precision drops > 0.02. Record before/after in the spec's
  results section / `DOWNLOAD.md`.

## Scope

**In:** `ReIDGallery`, its wiring into `find_best_match_reid` and the
tracking-time admission, config params, tests, LaSOT validation.

**Out (explicitly):** active call-out / re-seed (Spec B); removing the
terminal-LOST cap (deferred — see gap-analysis memory); depth-assisted
reacquisition (deferred, needs Orbbec bags); any threshold changes.

## Open questions / risks

- Max-over-views is the main precision-sensitive choice; the ±0.01 LaSOT
  guardrail + `score_mode` fallback are the mitigation. If LaSOT shows a
  precision dip that `top2_mean` doesn't fix, stop and reassess before merge.
