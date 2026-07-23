# kimi_api

LLM/VLM-backed person and scene action servers for Tinker (ROS2 Humble).

## Nodes

| Node | Interface | Purpose |
|---|---|---|
| `feature_recognition` | actions `/feature_extraction_service`, `/seat_recommend_service` | Describe the person addressing the robot (five slots: hair color, gender, age, glasses, upper-body wear) + legacy seat recommendation |
| `feature_matching` | action `/feature_matching_service` | Locate previously described/photographed people among detected persons (image mode when one reference image per feature is supplied; text-only fallback when fewer reference images than features are supplied; more references than features is rejected) |
| `grocery_categorize` | action `/grocery_categorize` | Categorize grocery items |
| `seat_recommend_bbox` | action `/seat_recommend_bbox_service` | VLM seat recommendation (bbox_select strategy) |
| `object_scan` | action `/object_scan` | Return the visible subset of a caller-provided object vocabulary |
| `seat_fewshot_annotator` | (no ROS interface — local web UI) | Dev tool: annotate seat few-shot examples |

Providers/keys: Gemini via `OPENROUTER_API_KEY` (primary), Qwen via
`DASHSCOPE_API_KEY` (fallback where configured). See
`src/tk26_vision/CLAUDE.md` for build (tkbuild/build.sh) and venv setup.

## Changelog

- 2026-07-02 — Feature description cut to five slots (hair color [+optional
  length], gender, approximate age in words, glasses, upper-body garment
  color+type), prompt-only; matching prompts (both modes) now cite the same
  five attributes as evidence and drop body shape/posture. Spec:
  `docs/superpowers/specs/2026-07-02-feature-5slot-cut-design.md`.
- 2026-07-02 — Person selection for feature extraction is size-primary:
  candidates shorter than 0.75x the tallest survivor no longer compete on
  centering (fixes background-crowd latching at the arena door);
  centermost + depth tie-break retained among comparable-size candidates.
