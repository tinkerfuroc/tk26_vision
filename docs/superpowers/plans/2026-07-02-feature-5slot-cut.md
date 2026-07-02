# Feature Description 5-Slot Cut Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Constrain kimi_api's person feature description to exactly five attributes (hair color [+optional length], gender, approximate age, glasses, upper-body wear) via prompt-only changes, and align both feature-matching prompts to cite the same five attributes as evidence.

**Architecture:** Hoist the extraction sys-prompt to a module constant (`FEATURE_SYS_PROMPT`) and the two matching sys-prompts into one pure builder (`build_matching_sys_prompt(n_cand, n_feats, text_only)`) so prompt content is unit-testable; rewrite the prompt text per the spec. No srv, blackboard, BT, or logic changes — the feature stays one free-text string end-to-end.

**Tech Stack:** Python 3.10, ROS2 Humble, pytest (venv `.venv-vision-main`), tkbuild deploy wrapper.

**Spec:** `src/tk26_vision/docs/superpowers/specs/2026-07-02-feature-5slot-cut-design.md` (committed, a813305)

## Global Constraints

- All commands run from workspace root `/home/tinker/tk25_ws`.
- Tests run with the vision venv python after sourcing ROS:
  `source /opt/ros/humble/setup.zsh && source install/setup.zsh && src/tk26_vision/.venv-vision-main/bin/python -m pytest …`
  (module imports need rclpy/cv_bridge/tinker_vision_msgs_26; `test/conftest.py` already shims the src tree ahead of the installed copy).
- NEVER run plain `colcon build`. Deploy = `tkbuild tk26_vision --packages-select kimi_api` (the runtime root install at `install/kimi_api` is a plain copy; source edits are invisible to `ros2 run` until rebuilt).
- ⚠️ **Concurrent-committer guard:** `kimi_api/kimi_api/feature_recognition.py` and `kimi_api/kimi_api/feature_matching.py` carry unrelated UNCOMMITTED in-flight changes (VLM provider-chain rewrite + the 2026-07-02 person-select fix) from another session. **Before any `git commit` that stages either file, ask the user** whether to sweep that in-flight work into the commit or defer source-file commits to the owning session. Committing NEW files (tests, README, docs) is always safe. Never `--amend`, never rebase; re-verify `git status` immediately before each commit (HEAD moves under you in this repo).
- Do not modify: srv definitions, `TemplateNodes/Vision.py`, any tk25_decision file, `select_best_person_idx`, the provider-chain modules (`_feature_vlm.py`, `_match_vlm.py`), or the forced-match/JSON-output contract inside the matching prompts.
- The extraction sentence is spoken verbatim by the Receptionist (`customNodes.py:148`) — the template must remain a natural English sentence.

---

### Task 1: Extraction prompt → 5-slot template

**Files:**
- Modify: `src/tk26_vision/src/kimi_api/kimi_api/feature_recognition.py` (sys_prompt at ~line 376 inside `feature_extraction_srv_callback`)
- Test: `src/tk26_vision/src/kimi_api/test/test_prompts_5slot.py` (create)

**Interfaces:**
- Produces: module constant `kimi_api.feature_recognition.FEATURE_SYS_PROMPT: str` (used by the callback and by tests; Task 2 does NOT depend on it).

- [ ] **Step 1: Write the failing tests**

Create `src/tk26_vision/src/kimi_api/test/test_prompts_5slot.py`:

```python
"""Prompt-content locks for the 2026-07-02 five-slot feature cut.

Spec: docs/superpowers/specs/2026-07-02-feature-5slot-cut-design.md.
The description is one free-text string end-to-end; these tests pin the
five requested attributes (hair color [+optional length], gender,
approximate age, glasses, upper-body wear) and the removal of the old
multi-clothing / facial-features asks, without asserting the full prompt
verbatim (wording may be tuned; slots may not drift).
"""
from kimi_api.feature_recognition import FEATURE_SYS_PROMPT


def test_extraction_prompt_requests_exactly_five_slots():
    for term in (
        'gender',
        'age in years',
        'hair color',
        'glasses',
        'upper-body garment',
    ):
        assert term in FEATURE_SYS_PROMPT, f'missing slot ask: {term}'
    # Old prompt asks that must be gone:
    assert 'pieces of clothing' not in FEATURE_SYS_PROMPT
    assert 'facial features' not in FEATURE_SYS_PROMPT


def test_extraction_prompt_keeps_spoken_sentence_template():
    # The Receptionist speaks this sentence verbatim (customNodes.py:148).
    assert 'years old' in FEATURE_SYS_PROMPT
    assert 'wearing a [color] [garment]' in FEATURE_SYS_PROMPT
    # Age-in-words convention retained:
    assert 'not numeric numerals' in FEATURE_SYS_PROMPT


def test_extraction_prompt_excludes_everything_else():
    assert 'Do not mention lower-body clothing' in FEATURE_SYS_PROMPT
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
source /opt/ros/humble/setup.zsh && source install/setup.zsh && \
src/tk26_vision/.venv-vision-main/bin/python -m pytest \
  src/tk26_vision/src/kimi_api/test/test_prompts_5slot.py -v
```
Expected: collection ERROR for the whole file — `ImportError: cannot import name 'FEATURE_SYS_PROMPT'` (the constant does not exist yet).

- [ ] **Step 3: Hoist the constant and write the new prompt**

In `feature_recognition.py`, add a module-level constant directly below the existing `HEIGHT_TIE_FRAC` definition:

```python
# Feature-description prompt, cut to five slots on 2026-07-02 (spec:
# docs/superpowers/specs/2026-07-02-feature-5slot-cut-design.md). The
# output sentence is spoken verbatim by the Receptionist introduction and
# is matched against candidate crops by feature_matching — keep it a
# natural sentence and keep the slots in sync with
# feature_matching.MATCH_EVIDENCE.
FEATURE_SYS_PROMPT = (
    'You will be asked to extract features of one single designated person in an image.'
    ' Describe EXACTLY these five attributes and nothing else: (1) gender, (2) approximate'
    ' age in years (give in words, such as "twenty", not numeric numerals), (3) hair color'
    ' (you may qualify it with hair length, such as "short black hair"), (4) whether the'
    ' person is wearing glasses or not, and (5) the most prominent upper-body garment as'
    ' color plus garment type (such as "red shirt"; if a jacket or coat covers the shirt,'
    ' describe the jacket). Output in the format of "[gender pronoun] is [gender],'
    ' approximately [age in words] years old, has [hair color] hair, is [wearing glasses'
    ' | not wearing glasses], and is wearing a [color] [garment]".'
    ' Do not mention lower-body clothing, shoes, accessories, or any other information.'
)
```

Then in `feature_extraction_srv_callback`, replace the inline prompt:

```python
        sys_prompt = (
            'You will be asked to extract features of one single designated person in an image,'
            ' including their gender, approximate age in years, facial features (hair length,'
            ' with or without glasses), hair color, and atleast two pieces of clothing (the more'
            ' the better, but no more than five). Output in the format of "[gender pronoun] is'
            ' [gender], [gender pronoun] are approximately [approximate age in years (give in'
            ' words, such as "twenty", not numeric numerals)] years-old, [gender pronoun] has'
            ' [hair color] hair and [facial features]. [gender pronoun] is wearing [clothing]",'
            ' do not include other information'
        )
```

with:

```python
        sys_prompt = FEATURE_SYS_PROMPT
```

- [ ] **Step 4: Run tests to verify they pass**

Run the same pytest command as Step 2.
Expected: 3 passed.

- [ ] **Step 5: Regression — person-select + provider-chain tests still green**

Run:
```bash
source /opt/ros/humble/setup.zsh && source install/setup.zsh && \
src/tk26_vision/.venv-vision-main/bin/python -m pytest \
  src/tk26_vision/src/kimi_api/test/test_feature_person_select.py \
  src/tk26_vision/src/kimi_api/test/test_feature_vlm.py -q
```
Expected: all passed (8 person-select + the provider-chain suite), no new failures.

- [ ] **Step 6: Commit (guarded)**

`git status` first. The new test file is always safe to commit. `feature_recognition.py` carries unrelated in-flight work (see Global Constraints) — **ask the user** before staging it; if deferred, commit the test file only and note the pending source hunk in the final report.

```bash
cd src/tk26_vision && git status --short src/kimi_api/ && \
git add src/kimi_api/test/test_prompts_5slot.py && \
git commit -m "test(kimi_api): prompt-content locks for 5-slot feature cut" \
  -- src/kimi_api/test/test_prompts_5slot.py
```

---

### Task 2: Matching prompts → same five attributes as evidence

**Files:**
- Modify: `src/tk26_vision/src/kimi_api/kimi_api/feature_matching.py` (inline sys_prompt construction at ~lines 325–352)
- Test: `src/tk26_vision/src/kimi_api/test/test_prompts_5slot.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1 (independent; can run in parallel).
- Produces: `kimi_api.feature_matching.build_matching_sys_prompt(n_cand: int, n_feats: int, text_only: bool) -> str` and module constant `kimi_api.feature_matching.MATCH_EVIDENCE: str`.

- [ ] **Step 1: Append the failing tests**

Append to `src/tk26_vision/src/kimi_api/test/test_prompts_5slot.py`:

```python
from kimi_api.feature_matching import build_matching_sys_prompt


def test_matching_prompt_text_only_cites_five_slots_not_body_shape():
    p = build_matching_sys_prompt(5, 3, True)
    for term in ('hair color', 'glasses', 'apparent age', 'upper-body'):
        assert term in p, f'missing evidence term: {term}'
    assert 'body shape' not in p
    assert 'posture' not in p
    # Structural contract unchanged:
    assert '(0..4)' in p
    assert 'length 3' in p
    assert 'EVERY description MUST be matched' in p
    assert 'NEVER use -1' in p


def test_matching_prompt_image_mode_cites_five_slots_not_posture():
    p = build_matching_sys_prompt(4, 2, False)
    for term in ('hair color', 'glasses', 'apparent age', 'upper-body'):
        assert term in p, f'missing evidence term: {term}'
    assert 'body shape' not in p
    assert 'posture' not in p
    # Structural contract unchanged:
    assert 'SAME' in p
    assert 'length 2' in p
    assert 'EVERY reference MUST be matched' in p
    assert 'tiebreaker hint' in p
    assert 'NEVER use -1' in p
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
source /opt/ros/humble/setup.zsh && source install/setup.zsh && \
src/tk26_vision/.venv-vision-main/bin/python -m pytest \
  src/tk26_vision/src/kimi_api/test/test_prompts_5slot.py -v
```
Expected: collection ERROR for the whole file — `ImportError: cannot import name 'build_matching_sys_prompt'` (the top-of-file import fails before any test runs; the Task-1 tests error too — that is expected at this step).

- [ ] **Step 3: Hoist the builder and rewrite the evidence lists**

In `feature_matching.py`, add at module level (below the existing imports/constants, above the node class):

```python
# Match-evidence attribute list — keep in sync with the five slots
# feature_recognition.FEATURE_SYS_PROMPT asks for (spec:
# docs/superpowers/specs/2026-07-02-feature-5slot-cut-design.md).
MATCH_EVIDENCE = (
    'hair color and length, gender, apparent age, glasses, and '
    'upper-body clothing color and type'
)


def build_matching_sys_prompt(n_cand: int, n_feats: int, text_only: bool) -> str:
    """Sys-prompt for the match call; pure so tests can pin its content.

    ``text_only`` selects the legacy descriptions-only wording; otherwise
    the reference-image wording is produced. Both keep the JSON-list
    output contract and the forced-match rule unchanged.
    """
    if text_only:
        return (
            f'You will be shown {n_cand} CANDIDATE crops of people and {n_feats} '
            f'textual DESCRIPTIONS. For each description, output the candidate index '
            f'(0..{n_cand - 1}) whose person best matches that description. '
            f'Output ONLY a JSON list of length {n_feats}, e.g. "[0, 3, 1]". '
            'EVERY description MUST be matched to a candidate. If you are uncertain, '
            f'pick the candidate whose visible features ({MATCH_EVIDENCE}) '
            'are CLOSEST to the description. NEVER use -1 or any negative number. '
            'Multiple descriptions MAY map to the same candidate. '
            'Do not include explanations.'
        )
    return (
        f'You will be shown {n_feats} REFERENCE images of specific people, then '
        f'{n_cand} CANDIDATE crops taken from a wider scene. For each reference '
        f'(0..{n_feats - 1}), output the candidate index whose person is the SAME '
        f'individual as the reference. Use {MATCH_EVIDENCE} as evidence. '
        'The user may also provide a textual description per reference; treat it '
        'as a tiebreaker hint only. '
        f'Output ONLY a JSON list of length {n_feats}, e.g. "[0, 2, 1]". '
        'EVERY reference MUST be matched to a candidate. If you are uncertain, '
        f'pick the candidate whose features ({MATCH_EVIDENCE}, and the '
        'description) are CLOSEST to the reference. NEVER use -1 or any '
        'negative number. Do not include explanations.'
    )
```

Then in `feature_matching_srv_callback`, replace the whole inline `if text_only_mode: sys_prompt = (...) else: sys_prompt = (...)` block (currently ~lines 325–352, starting `n_cand = len(cropped_person_imgs)`) with:

```python
        n_cand = len(cropped_person_imgs)
        sys_prompt = build_matching_sys_prompt(n_cand, n_feats, text_only_mode)
```

- [ ] **Step 4: Run tests to verify they pass**

Same pytest command as Step 2.
Expected: 5 passed.

- [ ] **Step 5: Commit (guarded)**

Same guard as Task 1 Step 6: test-file changes commit freely; **ask the user** before staging `feature_matching.py` (in-flight work).

```bash
cd src/tk26_vision && git status --short src/kimi_api/ && \
git add src/kimi_api/test/test_prompts_5slot.py && \
git commit -m "test(kimi_api): matching-prompt evidence locks for 5-slot cut" \
  -- src/kimi_api/test/test_prompts_5slot.py
```

---

### Task 3: Package suite, README + changelog, deploy, live-check handoff

**Files:**
- Create: `src/tk26_vision/src/kimi_api/README.md`
- No source modifications.

**Interfaces:**
- Consumes: Tasks 1–2 landed in the working tree.
- Produces: deployed runtime install + operator checklist; user-facing README with changelog.

- [ ] **Step 1: Run the full kimi_api suite**

Run:
```bash
source /opt/ros/humble/setup.zsh && source install/setup.zsh && \
src/tk26_vision/.venv-vision-main/bin/python -m pytest \
  src/tk26_vision/src/kimi_api/test/ -q 2>&1 | tail -5
```
Expected: only the three pre-existing package-wide failures (`test_copyright`, `test_flake8`, `test_pep257` — over-long legacy lines and D213 docstring style, documented 2026-07-02); everything else passes. Any NEW failure = stop and fix before proceeding.

- [ ] **Step 2: Create the package README with changelog**

Create `src/tk26_vision/src/kimi_api/README.md`:

```markdown
# kimi_api

LLM/VLM-backed person and scene services for Tinker (ROS2 Humble).

## Nodes

| Node | Interface | Purpose |
|---|---|---|
| `feature_recognition` | `/feature_extraction_service`, `/seat_recommend_service` | Describe the person addressing the robot (five slots: hair color, gender, age, glasses, upper-body wear) + legacy seat recommendation |
| `feature_matching` | `/feature_matching_service` | Locate previously described/photographed people among detected persons (image mode when one reference image per feature is supplied; text-only fallback otherwise) |
| `grocery_categorize` | action `/grocery_categorize` | Categorize grocery items |
| `seat_recommend_bbox` | `/seat_recommend_bbox` | VLM seat recommendation (bbox_select strategy) |

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
```

- [ ] **Step 3: Deploy to the runtime tree**

Run:
```bash
tkbuild tk26_vision --packages-select kimi_api 2>&1 | tail -3
grep -c 'FEATURE_SYS_PROMPT' \
  /home/tinker/tk25_ws/install/kimi_api/lib/python3.10/site-packages/kimi_api/feature_recognition.py
grep -c 'build_matching_sys_prompt' \
  /home/tinker/tk25_ws/install/kimi_api/lib/python3.10/site-packages/kimi_api/feature_matching.py
```
Expected: build finishes; both greps return >= 1 (the root-install copy — the tree `ros2 run` uses — carries the new prompts).

- [ ] **Step 4: Commit README + plan doc (guarded for source files)**

```bash
cd src/tk26_vision && git status --short && \
git add src/kimi_api/README.md docs/superpowers/plans/2026-07-02-feature-5slot-cut.md && \
git commit -m "docs(kimi_api): README + changelog; 5-slot cut implementation plan" \
  -- src/kimi_api/README.md docs/superpowers/plans/2026-07-02-feature-5slot-cut.md
```

Then **ask the user once** about the two source files (`feature_recognition.py`, `feature_matching.py`): commit now (sweeps the concurrent session's in-flight VLM-chain + person-select work into the commit) or leave for the owning session. Follow their answer.

- [ ] **Step 5: Operator live-check handoff (report, do not execute)**

Report to the user (nodes run under their launch; restart is operator-driven):
1. Restart the kimi_api nodes (running instances predate the rebuild).
2. One extraction call → newest `vision_log/<session>/feature_service_*_feature_extraction_req_*.json` `feature` field shows the 5-slot sentence (and nothing about lower-body clothing).
3. One matching call (hri-2026 flow) → newest `feature_matching_service_*_req_*.json` shows `"text_only_mode": false` with one `ref<i>` image per feature and the new-style `features_text`.
4. Re-author the host's on-disk description file (the one `BtNode_LoadPersonReference` loads) in the 5-slot sentence style so the pre-registered host reference matches the new format.
