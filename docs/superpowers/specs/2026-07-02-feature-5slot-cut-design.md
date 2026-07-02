# Feature description cut to 5 slots (prompt-only)

**Date:** 2026-07-02
**Status:** Approved (design reviewed in-session)
**Packages:** `kimi_api` (code), `tk25_decision/behavior_tree` (verification only)

## Problem

The feature-extraction VLM prompt currently requests gender, approximate age,
facial features (hair length, glasses), hair color, and two-to-five pieces of
clothing. The resulting long descriptions add noise to feature matching at
crowded venues (lower-body clothing and body-shape cues are frequently
occluded for seated guests), and the matching prompts tell the VLM to weigh
evidence ("clothing, hair color/length, body shape, posture") that doesn't
line up with what extraction reliably produces.

## Goal

Constrain the description to exactly five attributes, end-to-end as one
free-text sentence (no interface changes):

1. hair color (hair length MAY be mentioned alongside, e.g. "short black hair")
2. gender
3. approximate age
4. glasses / no glasses
5. upper-body wear as color + garment (e.g. "red shirt")

## Non-goals

- No structured/JSON feature schema; `FeatureExtraction.srv` /
  `FeatureMatching.srv` and the BT blackboard contract stay untouched.
- No output validation or retry-on-malformed-description (prompt-only; the
  provider chain's existing error retries are unchanged).
- No change to the forced-match behavior in `feature_matching` ("EVERY
  reference MUST be matched") — that abstain-option fix is a separate,
  already-flagged follow-up.
- No change to person selection (`select_best_person_idx`) — fixed separately
  on 2026-07-02.

## Design

### 1. Extraction prompt (`kimi_api/feature_recognition.py`, `sys_prompt`)

Request exactly the five attributes. Output template (speech-friendly — the
Receptionist speaks this sentence verbatim during introductions):

> "[pronoun] is [gender], approximately [age in words] years old, has
> [hair color] hair, is [wearing glasses | not wearing glasses], and is
> wearing a [color] [garment]."

Prompt rules:
- age in words ("twenty"), not numerals (existing convention, kept);
- upper-body wear = the most prominent visible upper-body garment with its
  color; if a jacket/coat covers the shirt, describe the jacket;
- hair length is optional and may qualify hair color ("short gray hair");
- explicitly: do not mention lower-body clothing, shoes, accessories, or
  anything else.

### 2. Matching prompts (`kimi_api/feature_matching.py`, both modes)

Replace the evidence enumerations:
- text-only mode: "clothing, hair, body shape" → "hair color/length, gender,
  apparent age, glasses, and upper-body clothing color/type";
- image mode: "clothing, hair color/length, body shape, and posture" →
  "hair color/length, gender, apparent age, glasses, and upper-body clothing
  color/type" (reference images remain the primary evidence; the textual
  description remains a tiebreaker hint, as today).

Everything else in both prompts (JSON-list output contract, forced-match
rule, index ranges) is unchanged.

### 3. Untouched surfaces (verified consumers, all opaque-string passthroughs)

| Consumer | Location | Behavior with new format |
|---|---|---|
| Blackboard store | `TemplateNodes/Vision.py:676` (`BtNode_FeatureExtraction`) | passthrough |
| Matching request | `TemplateNodes/Vision.py:1047` (`BtNode_FeatureMatching`) | passthrough |
| Seat recommendation requests | `TemplateNodes/Vision.py:828,931` | passthrough |
| Spoken introduction | `Receptionist/customNodes.py:148` ("You will meet guest <name>. <features>") | new sentence is still natural speech |
| Host pre-registration | `BtNode_LoadPersonReference` (`Vision.py:765`, disk file) | see operational note |

**Operational note (not code):** the host's on-disk description file loaded by
`BtNode_LoadPersonReference` should be re-authored on the robot in the 5-slot
style so the host reference matches the new format.

### 4. hri-2026 image-mode confirmation (verification finding, no change)

`createHRITask2026` → `hri.createTwoWayIntroduction()` →
`BtNode_FeatureMatching(trim_last_person=False)`. The BT always sends exactly
one `comparison_image` per person and `feature_matching` selects image mode
when ref count == feature count, so hri-2026 runs image mode structurally;
confirmed live 2026-07-02 09:18:54 (`"text_only_mode": false`, 3 refs) in
`vision_log/20260702_070449`.

**Documented latent fragility (pre-existing, out of scope):** if a `Person`
ever carries `comparison_image is None`, `BtNode_FeatureMatching` substitutes
an empty `Image()`, which still triggers image mode with a degenerate
reference. Today every person always gets a real crop (guests via extraction,
host via LoadPersonReference).

## Testing & verification

1. `kimi_api` pytest suite stays green (no test asserts prompt content; no
   test edits expected).
2. Rebuild + deploy: `tkbuild tk26_vision --packages-select kimi_api`
   (runtime root install is a plain copy), restart the kimi_api nodes.
3. Live behavioral check (operator): one extraction call — the vision_log
   `feature` field shows the 5-slot sentence; one matching call — req JSON
   shows `text_only_mode: false` and the new-style `features_text`.
