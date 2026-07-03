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
**hardcoded default `'dashscope'`, no environment variable involved**)
declared on every node at the 11 call sites:

```python
self.declare_parameter('qwen_api_backend', 'dashscope')
```

**No env var, no `.env`, no shell/robot-profile involvement.** An earlier
draft tried to source the default from a `QWEN_API_BACKEND` env var so one
export would flip a whole robot. Two rounds of review found this
unworkable: `ros2 launch` never loads `.env` (only each node's own
`load_dotenv()` call does), so a launch-time default sourced from `.env`
silently diverges from a node-time default sourced the same way, and the
fix for that (shell-exporting the var instead, e.g. via
`tinker_robot_config`/`robot-env.sh`) was rejected. Per explicit direction:
`qwen_api_backend` is a **plain launch/ROS parameter with no smart
default** — the operator (or whatever per-robot script invokes
`ros2 launch` / `ros2 run`) passes `qwen_api_backend:=openrouter` (or
`-p qwen_api_backend:=openrouter` for a bare `ros2 run`) explicitly, every
time, for the robot that needs it. Consistency across a robot's 11 call
sites is the responsibility of whatever wraps the launch commands for that
robot (e.g. hardcoded in that robot's launch invocation), not this design's
mechanism — it deliberately owns nothing beyond "one ROS parameter per
node, no magic default source."

**Launch-file wiring for the 5 nodes `vision_bringup.launch.py` manages**
(`generalist_node`, `waving_person_server`, `feature_recognition`,
`feature_matching`, `seat_recommend_bbox`): a plain
`DeclareLaunchArgument('qwen_api_backend', default_value='dashscope')`,
passed straight through as a `-p` override — no substitution logic, no env
lookup, nothing to get subtly wrong.

**Known gap:** the other 4 node executables (`grocery_categorize`,
`object_match_server`, `object_match_all_server`, `placing_location_server`)
have no launch file at all today (only bare `ros2 run`, considered obsolete
generally but out of scope to fix here). They get the same
`declare_parameter('qwen_api_backend', 'dashscope')` and must be flipped via
an explicit `-p` override on their own `ros2 run` invocation; giving them
proper launch-file coverage is a separate follow-up, not bundled here.

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

**Note — this overrides a documented convention.** `_waving_vlm.py` (and,
per `tk26_vision/CLAUDE.md`, `qwen_match_vlm.py`/`vlm_match_client.py` by
the same convention) explicitly document staying `kimi_api`-free as an
intentional decoupling choice. This design knowingly overrides that
convention for one shared function, for the reason above. The
implementation must update those modules' docstrings and the CLAUDE.md
line that describes the convention — leaving stale docs claiming
decoupling that no longer holds would confuse the next person touching
these files.

`generalist_node.py` / `vlm_bbox.py` do **not** use the shared resolver —
they keep their existing `dashscope/model`-prefix convention
(`_split_provider`). Two DashScope-pointing defaults are involved:

- `dashscope_qwen_model` (default `'dashscope/qwen3-vl-plus'`, consulted
  only when `prefer_dashscope_qwen=True`)
- `vlm_fallback_models` (default `['dashscope/qwen3-vl-plus']` — this is the
  **actual default-path** fallback; an earlier draft of this design missed
  it entirely, which would have left the default path pinned to DashScope
  even with the toggle flipped)

**No auto-flip for either.** An earlier draft proposed automatically
rewriting `dashscope/`-prefixed entries to OpenRouter model strings when
`qwen_api_backend='openrouter'`. Review rejected this: detecting "this list
is still at its unmodified default" by comparing it against the known
default value is the exact same string(list)-comparison ambiguity already
rejected for the scalar model params below (an operator who explicitly sets
`vlm_fallback_models: ['dashscope/qwen3-vl-plus']` on purpose would get it
silently rewritten). Applying that already-rejected pattern to a list
doesn't make it safer. Instead: **fail fast at node `__init__`** if
`qwen_api_backend='openrouter'` and any of the following is a
`dashscope/`-prefixed value: `dashscope_qwen_model` (when
`prefer_dashscope_qwen=True`), any entry in `vlm_fallback_models` (when
false), **or the primary `vlm_model` param itself** if an operator
explicitly pointed it at a `dashscope/…` model — the first two drafts of
this check only covered the two fallback-path params and missed that
`vlm_model` (default `'google/gemini-2.5-flash'`) can independently be set
to a `dashscope/…` value. Error tells the operator to pass an explicit
OpenRouter-pointing value for whichever param is active. This costs the
operator one extra explicit override on this node when flipping the
toggle; the alternative is reintroducing a silent-rewrite bug on a list.

This check only fires under an explicit `qwen_api_backend='openrouter'` —
`_split_provider` (`vlm_bbox.py`) still routes **per-entry**, so a
deliberate mixed-provider fallback chain (e.g.
`['dashscope/qwen3-vl-plus', 'openai/gpt-4o']`) remains fully expressible
and untouched under the default `qwen_api_backend='dashscope'`. The
fail-fast only blocks the specific combination of "this robot declared
no-DashScope" plus "this param still points at DashScope" — it doesn't
remove per-entry routing generally.

`prefer_dashscope_qwen` as a param name becomes slightly misleading under
this toggle (it now means "prefer the qwen leg over gemini", independent of
which backend serves that leg) — rename is optional, not required; flag for
the implementation plan to decide.

## Model substitution — sentinel default, not string comparison

The earlier draft tried to detect "operator left the model param at its
default" by comparing the current value to the node's hardcoded default
string (`'qwen3-vl-plus'`). This breaks when an operator explicitly sets the
param to that same value on purpose (e.g. for config clarity) — it would be
silently rewritten to a different model with no error.

**Fix:** 7 of the per-node qwen-model params currently default to
`'qwen3-vl-plus'`: `feature_model_qwen` (`feature_recognition.py`),
`match_model_qwen` (`feature_matching.py`), `categorize_model_qwen`
(`grocery_categorize.py`), `bbox_model_qwen` (`seat_recommend_bbox.py`),
`placing_model_qwen` (`placing_location_server.py`), `vlm_model_qwen`
(`waving_person_server.py`), and `vlm_model` (`object_match_server.py` —
**not** `vlm_model_qwen`, different name than waving's despite the earlier
draft claiming a shared name). All 7 need their default changed to an
**empty-string sentinel** (`''`). Note `object_match_server`'s `vlm_model`
shares its literal name with `object_match_all_server.py`'s `vlm_model` —
same string, different node, different current default (the latter already
defaults to `''`) — worth a one-line comment in each file so the name
collision doesn't read as a copy-paste bug later.
`object_match_all_server.py`'s `vlm_model`/`judge_model` already default to
`''` — no change needed there, already sentinel-shaped. `resolve_qwen_target`
treats `''` as "use the backend's own default model" and honors any
non-empty value verbatim, on either backend. This removes the ambiguity
class entirely instead of working around it.

Two docs currently describe the old `'qwen3-vl-plus'` default and must be
updated in the same change: `tk26_vision/CLAUDE.md`'s per-node param table,
and `kimi_api/.env.example`'s comments if it references a model default.

**Init-time validation:** `resolve_qwen_target` (or its caller) checks the
model-id shape against the selected backend — OpenRouter ids contain `/`
(`org/name`), DashScope ids don't — and raises at node `__init__` on a
mismatch (e.g. a bare DashScope id explicitly set while
`backend='openrouter'`), rather than letting it reach a live API call and
fail with an opaque 404 mid-task.

**Interim behavior before the benchmark task (below) lands a real
default:** `resolve_qwen_target` must raise at `__init__` if
`backend='openrouter'` and the resolved OpenRouter default-model constant
is unset/empty — not silently proceed with a blank model string. In
practice the implementation plan should sequence the benchmark task before
any node ships with `qwen_api_backend='openrouter'` reachable in
production, but this is a defensive backstop, not a substitute for that
ordering.

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
   verified against a known-position target per `vlm_bbox.py:70-76`).
   **Only `vlm_bbox.py` switches decoders by model family** — the other
   five bbox-producing modules (`_seat_bbox_vlm.py`, `qwen_match_vlm.py`,
   `vlm_match_client.py`, `vlm_judge_client.py`, and whatever
   `waving_person_server`/`placing_location_server` use) hardcode a
   0-1000-normalized decode unconditionally, with no model check at all.
   Picking an OpenRouter model whose output format doesn't match produces
   **silently wrong coordinates on those five, not an error** — there is
   no format-mismatch detection to catch it. The benchmark task must
   either (a) confirm the chosen model matches the calibrated format on
   all six consumers before it ships, or (b) if a chosen model needs a
   different decode, extend the remaining five modules with the same
   family-aware branching `vlm_bbox.py` already has — this is real
   implementation work the benchmark task's results determine the size of,
   not a guaranteed drop-in.
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
- **Fail-fast applies only where the VLM call is the sole serving path —
  not merely wherever Qwen is nominally "primary."** The earlier draft's
  rule ("Qwen primary → fail fast") is wrong for `waving_person_server`:
  it defaults `vlm_provider='qwen'` (primary within its *VLM-only* chain),
  but the node's actual documented contract is graceful-disable — a
  missing key today does not crash it, because MediaPipe still serves
  waving detection without any VLM at all. The earlier rule would turn
  that into an init crash and break an existing no-key-startup guarantee
  (part of the T1 test tier). Corrected rule: fail fast only where there
  is no non-VLM fallback at all for that feature (e.g.
  `seat_recommend_bbox`'s `bbox_select` strategy has no non-VLM seat
  detector). Where a non-VLM fallback exists — `waving_person_server`
  (MediaPipe) — preserve today's graceful-disable regardless of which leg
  is "primary" within the VLM-only chain. `object_match_all_server` is
  **confirmed to have no non-VLM fallback** (it's purely VLM-driven
  matching/segmenting) and **already fails fast today** on a missing key
  (`vlm_match_client.py`'s client raises `RuntimeError` at construction) —
  so fail-fast for it is both correct and behavior-preserving, not a new
  restriction. This must still be checked per node during implementation
  for any site not named here, not assumed from the `vlm_provider` default
  alone.
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

## Other noted items

- **`object_match_all_server.py` has an existing `vlm_base_url` param**
  that overlaps with the new toggle's base-URL selection. Precedence
  between an explicit `vlm_base_url` override and the resolver's
  backend-derived URL is undecided — recommend explicit `vlm_base_url`
  wins (consistent with "explicit values are always honored verbatim"
  elsewhere in this design), but this needs a one-line decision in the
  implementation plan, not left implicit.
- **Match/judge backend consistency in `object_match_all_server` holds
  only by construction** — both `vlm_match_client.py` and
  `vlm_judge_client.py` are fed the same single node-level
  `qwen_api_backend` param, so they can't diverge. Stating this
  explicitly so it isn't accidentally broken by a future per-call
  override.
- **Accepted tradeoff, not a defect:** setting `qwen_api_backend=
  'openrouter'` collapses both chain legs (Gemini and Qwen) onto one
  gateway, losing the host-independence `vlm_bbox.py` currently
  documents as deliberate (`vlm_bbox.py:296-300`). This is the intended
  effect of a per-robot backend choice driven by regional latency, not an
  oversight — noted so it isn't rediscovered as a "bug" later.

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

Two rounds of adversarial review by an independent subagent
(model: `claude-fable-5`), each verifying claims against the live codebase
rather than reasoning about the design in the abstract.

**Round 1** found and this version incorporates fixes for: sentinel-default
model detection (was string-comparison), the `vlm_fallback_models` gap in
`generalist_node` (original draft only touched `dashscope_qwen_model`), the
factually-incorrect "new coupling" rationale for a separate
`tk_vision_specialized` resolver, backend-aware fallback-arming, and the
model-choice methodology (deferred to a benchmark task instead of picking
`qwen3.7-plus` on cost/recency alone).

**Round 2** re-verified round 1's fixes against the code and found: the
round-1 launch-arg-vs-env-default fix didn't actually work (`ros2 launch`
never loads `.env`, so the env-sourced launch default silently diverged
from the node-time default) — resolved by dropping the env-var mechanism
entirely per explicit direction, `qwen_api_backend` is now a plain
parameter with a hardcoded default and no smart source; the fail-fast rule
would have regressed `waving_person_server`'s existing no-key
graceful-disable behavior — corrected to key on "no non-VLM fallback
exists" rather than "Qwen is nominally primary"; the `generalist_node`
list-flip was underspecified in a way that could silently rewrite explicit
operator config — corrected to fail-fast instead of auto-flip, for
consistency with the sentinel-default principle; only `vlm_bbox.py` has
model-family-aware bbox decoding, the other five bbox-producing modules
hardcode the format with no mismatch detection — now called out as
real implementation work the benchmark task's results size, not a
guaranteed drop-in; plus a sentinel-count correction (7 params, not 9;
`object_match_all_server` was already sentinel-shaped) and several minor
doc/precedence items.

**Round 3** verified round 2's fixes hold and returned "approve with minor
fixes": corrected `object_match_server`'s sentinel param name (`vlm_model`,
not the previously-claimed shared `vlm_model_qwen`) and flagged its name
collision with `object_match_all_server`'s already-sentinel `vlm_model`;
resolved the `object_match_all_server` fail-fast hedge definitively (no
non-VLM fallback exists, and it already fails fast today on a missing key
— confirmed, not assumed); extended the `generalist_node` fail-fast check
to also cover an explicitly `dashscope/`-prefixed primary `vlm_model`
(the prior two drafts checked only the two fallback-path params); and
added an explicit note that the fail-fast doesn't disable legitimate
mixed-provider fallback chains under the default backend, only the
declared-no-DashScope-but-still-pointed-at-DashScope combination.
