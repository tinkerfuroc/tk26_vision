# Qwen VLM backend toggle: DashScope vs OpenRouter

**Date:** 2026-07-03
**Status:** Design approved, pending implementation plan.

## Motivation

Tinker robots deploy to physically distant locations (one in Korea, one in
Canada). DashScope's fixed regional endpoints (China-mainland default, or an
international `dashscope-intl` host) may not be the lowest-latency choice for
both. OpenRouter routes to whatever inference provider it selects, which may
beat a fixed DashScope region for at least one deployment. This is framed as
a **per-robot infrastructure choice**, not a per-request or per-feature one —
the whole robot should use one backend, consistently, for the duration of a
run.

## Scope

All 11 Qwen VLM call sites across three packages:

- `kimi_api` (5): `_seat_vlm.py`, `_seat_bbox_vlm.py`, `_match_vlm.py`,
  `_feature_vlm.py`, `_categorize_vlm.py`
- `tk_vision_specialized` (5): `qwen_match_vlm.py`, `vlm_match_client.py`,
  `vlm_judge_client.py`, `_waving_vlm.py`, `placing_vlm.py`
- `object_detection_generalist` (1): `vlm_bbox.py` — already has partial
  dual-host support via a `dashscope/model`-prefix convention.

**Explicitly out of scope:** consolidating the duplicated Gemini-primary →
Qwen-fallback chain-building logic (reimplemented per-node across ~6 files),
and fixing the pre-existing `DASHSCOPE_API_KEY`/`DASHCOPE_API_KEY` env-var
order inconsistency across 4 modules. Both are adjacent tech debt, left
untouched in this pass.

**Deliberately excluded (not call sites needing this toggle):**
`kimi_api/_vlm_text.py` (shared plumbing, DashScope mentioned only in a
comment), `seat_bench/providers.py` and
`scripts/produce_match_ground_truth.py` (offline benchmarking/ground-truth
tooling that intentionally hardcodes DashScope). Noted here so a future pass
doesn't "fix" them inconsistently.

## Mechanism

New ROS parameter `qwen_api_backend` (`'dashscope'` | `'openrouter'`,
default `'dashscope'`) declared on every node at the 11 call sites. Its
declared default is sourced from a `QWEN_API_BACKEND` env var:

```python
self.declare_parameter(
    'qwen_api_backend',
    os.environ.get('QWEN_API_BACKEND', 'dashscope'),
)
```

**Durable home for the env var:** the workspace-root `.env` file. Every
affected node already calls `load_env()` (python-dotenv, CWD-upward) at
startup before declaring parameters — this ordering is a hard requirement of
this design and must be preserved (`load_env()` before
`declare_parameter('qwen_api_backend', ...)`). Putting `QWEN_API_BACKEND=
openrouter` in the root `.env` makes the setting durable, discoverable, and
reachable identically whether a node is started via `vision_bringup`'s
launch file, a tmux script, or a future systemd unit — because the *node*
resolves it, not the launcher.

**Launch-file wiring for the 5 nodes `vision_bringup.launch.py` manages**
(`generalist_node`, `waving_person_server`, `feature_recognition`,
`feature_matching`, `seat_recommend_bbox`): a matching
`DeclareLaunchArgument('qwen_api_backend', ...)` whose default is the env
substitution, **not a literal string**:

```python
DeclareLaunchArgument(
    'qwen_api_backend',
    default_value=EnvironmentVariable('QWEN_API_BACKEND', default_value='dashscope'),
),
```

This is load-bearing: an explicitly-passed launch parameter overrides a
node's env-sourced `declare_parameter` default. If the launch arg defaulted
to the literal `'dashscope'`, it would silently override
`QWEN_API_BACKEND=openrouter` for exactly the 5 nodes that have launch-file
coverage — the opposite of intent.

**Known gap:** the other 4 node executables (`grocery_categorize`,
`object_match_server`, `object_match_all_server`, `placing_location_server`)
have no launch file at all today (only bare `ros2 run`, considered obsolete
generally but out of scope to fix here). They still honor
`QWEN_API_BACKEND` via their own `declare_parameter` default, so the toggle
works for them today; giving them proper launch-file coverage is a separate
follow-up, not silently bundled into this change.

## No credential changes

`OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` already exist in `.env` and are
already used today for the Gemini-primary leg of every chain. No new
secrets.

## Components

**One shared resolver**, added to `kimi_api/_env.py`:

```python
def resolve_qwen_target(backend: str, model_param_value: str) -> tuple[str, str, str]:
    """Return (base_url, api_key, model) for the selected Qwen backend.

    `model_param_value` is the node's own qwen-model ROS param. Empty string
    means "use the backend's default model"; any non-empty value is honored
    verbatim on either backend.
    """
```

Every one of the 11 call sites imports this — including
`tk_vision_specialized`'s five modules. The earlier plan to give
`tk_vision_specialized` its own separate copy was based on an "avoid new
coupling" rationale that doesn't hold: `tk_vision_specialized` already
depends on `kimi_api` (`package.xml` `exec_depend`) and already imports
`kimi_api._env` in `placing_vlm.py`. One resolver avoids a second copy of
the OpenRouter-default-model constant drifting out of sync — the same bug
class as the existing `DASHSCOPE_API_KEY`/`DASHCOPE_API_KEY` order split
that's already tech debt in this codebase.

`generalist_node.py` / `vlm_bbox.py` do **not** use the shared resolver —
they keep their existing `dashscope/model`-prefix convention
(`_split_provider`). The toggle must rewrite **both** of that node's
DashScope-pointing defaults, not just one:

- `dashscope_qwen_model` (consulted only when `prefer_dashscope_qwen=True`)
- `vlm_fallback_models` (default `['dashscope/qwen3-vl-plus']` — this is the
  **actual default-path** fallback; the earlier draft of this design missed
  it entirely, which would have left the default path pinned to DashScope
  even with the toggle flipped)

Both must flip their `dashscope/`-prefixed entries to bare OpenRouter model
strings when `qwen_api_backend='openrouter'`. `prefer_dashscope_qwen` as a
param name becomes slightly misleading under this toggle (it now means
"prefer the qwen leg over gemini", independent of which backend serves that
leg) — rename is optional, not required; flag for the implementation plan to
decide.

## Model substitution — sentinel default, not string comparison

The earlier draft tried to detect "operator left the model param at its
default" by comparing the current value to the node's hardcoded default
string (`'qwen3-vl-plus'`). This breaks when an operator explicitly sets the
param to that same value on purpose (e.g. for config clarity) — it would be
silently rewritten to a different model with no error.

**Fix:** each of the 9 per-node qwen-model params (`feature_model_qwen`,
`match_model_qwen`, `categorize_model_qwen`, `bbox_model_qwen`,
`vlm_model_qwen`, etc.) changes its default to an **empty-string sentinel**
(`''`). `resolve_qwen_target` treats `''` as "use the backend's own
default model" and honors any non-empty value verbatim, on either backend.
This removes the ambiguity class entirely instead of working around it.

**Init-time validation:** `resolve_qwen_target` (or its caller) checks the
model-id shape against the selected backend — OpenRouter ids contain `/`
(`org/name`), DashScope ids don't — and raises at node `__init__` on a
mismatch (e.g. a bare DashScope id explicitly set while
`backend='openrouter'`), rather than letting it reach a live API call and
fail with an opaque 404 mid-task.

## Default OpenRouter model — deferred, not picked yet

OpenRouter does not carry the exact `qwen3-vl-plus` slug this codebase
defaults to. Rather than guess, **the implementation plan must include a
benchmark/verification task before any default is locked in**, checking:

1. **Image modality** — does the candidate model actually accept image
   input on OpenRouter (confirm against OpenRouter's model metadata, not
   just its name).
2. **Bounding-box coordinate format** — `vlm_bbox.py` already has an
   empirically-calibrated decoder for `qwen3-vl`-family output (0-1000
   normalized xyxy, matched by a `'qwen3'` substring in the model name,
   verified against a known-position target per `vlm_bbox.py:70-76`). Any
   candidate model must be verified against that same
   known-position-target methodology before it's trusted for
   `seat_recommend_bbox`, `object_match_server`, `object_match_all_server`,
   `waving_person_server`, `placing_location_server`, or
   `generalist_node`'s bbox path — a wrong-format decode fails silently
   (wrong coordinates, not an error).
3. **Regional latency** — a timed round-trip check from (or toward) each
   deployment region, against DashScope-cn, DashScope-intl, and 2-3
   OpenRouter candidates. This is the feature's entire stated motivation;
   shipping a default with zero latency measurement defeats the purpose.

Candidates to benchmark: `qwen/qwen3-vl-32b-instruct` (open-weight, same
family as the calibrated bbox decoder — the safer starting bet on
correctness), `qwen/qwen3.6-plus` / `qwen/qwen3.7-plus` (proprietary
"Plus"-tier siblings, unverified on all three axes above but worth
measuring since they're the closer conceptual match to DashScope's hosted
tier).

## Error handling

- `backend='openrouter'` with `OPENROUTER_API_KEY` unset → fail fast at
  node `__init__`, mirroring today's `require_dashscope_api_key()` pattern.
- Invalid `qwen_api_backend` string (not `'dashscope'`/`'openrouter'`) →
  fail fast at init, never silently default.
- Model-id-shape-vs-backend mismatch (see above) → fail fast at init.
- **Fail-fast applies to the key required by the chain's primary leg.**
  Where Qwen is a *fallback* leg (most chains — Gemini primary), a missing
  key for the selected Qwen backend preserves today's graceful-disable
  behavior (skip the fallback, log a warning) rather than crashing the node
  — this matches existing behavior for the OpenRouter/Gemini key today.
  Where Qwen is the *primary* leg (`seat_recommend_bbox`'s `bbox_select`
  strategy, `vlm_provider='qwen'` default), a missing key for the selected
  backend fails fast, since there's no primary to fall back to.
- **Fallback-arming checks become backend-aware.** Several chains today
  arm their Qwen fallback only if `DASHSCOPE_API_KEY` is present. Under
  `backend='openrouter'`, that check must test for the key the *selected*
  backend needs (`OPENROUTER_API_KEY`, already required for the Gemini
  primary in every such chain, so this mostly comes out in the wash) —
  otherwise a robot configured for a pure-OpenRouter setup with no
  DashScope key would silently lose the Qwen fallback it was just routed to
  use. This touches per-node chain-arming conditionals; it is a necessary
  consequence of the toggle, not part of the out-of-scope chain
  consolidation.
- No live-swap: backend is fixed for the node's process lifetime.

## Testing

- Unit tests on `resolve_qwen_target` directly (pure function, no ROS
  graph): `dashscope` + sentinel model, `dashscope` + explicit override,
  `openrouter` + sentinel model, `openrouter` + explicit override, invalid
  backend (raises), model-shape-vs-backend mismatch (raises).
- Audit existing tests across all 11 call sites + `generalist_node` for
  hardcoded assumptions about `dashscope_base_url()` or `'qwen3-vl-plus'`
  defaults that would break silently once defaults change to the sentinel.
- No live network calls in CI (matches existing T0 static tier).
- **Smoke plan, strengthened:** not just one representative node — at
  least one known-position bbox verification per decoder consumer
  (`seat_recommend_bbox` bbox_select, `object_match_server`,
  `object_match_all_server`, `waving_person_server`,
  `placing_location_server`, `generalist_node`'s vlm_bbox path) on the
  OpenRouter backend, using the same known-position-target methodology
  `vlm_bbox.py` already uses for its DashScope-path calibration. A
  text-only smoke check (e.g. `grocery_categorize` alone) would pass while
  every bbox-producing site silently returns wrong geometry.

## Review history

Design was adversarially reviewed by an independent subagent
(model: `claude-fable-5`) against the actual codebase before this version.
That review found and this version incorporates fixes for: sentinel-default
model detection (was string-comparison), the `vlm_fallback_models` gap in
`generalist_node` (original draft only touched `dashscope_qwen_model`), the
launch-arg-vs-env-default precedence bug, the factually-incorrect
"new coupling" rationale for a separate `tk_vision_specialized` resolver,
backend-aware fallback-arming, and the model-choice methodology (deferred
to a benchmark task instead of picking `qwen3.7-plus` on cost/recency
alone).
