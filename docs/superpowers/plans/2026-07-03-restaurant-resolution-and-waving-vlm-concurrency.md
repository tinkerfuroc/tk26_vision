# Restaurant Orbbec Resolution + Concurrent Waving VLM Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch Restaurant's Orbbec at 1920x1080 (same launch-time-only pattern as HRI), and make `waving_person_server`'s VLM fallback actually run — concurrently with the CV pass, with an early exit once CV alone finds enough wavers — instead of being dead code behind a condition no caller ever satisfies.

**Architecture:** Two independent changes. (1) A one-file launch-script edit, identical in spirit to the HRI resolution work. (2) A restructuring of `waving_person_server.py`'s service callback: the VLM chain call launches on a background thread as soon as the synced RGB frame is available (concurrent with the CV/MediaPipe pass, which doesn't need it), and a new pure decision function determines whether to block on it or abandon it once CV's own results are in.

**Tech Stack:** ROS2 Humble, Python 3.10 (`tk26_vision/.venv-vision-main`), `concurrent.futures.ThreadPoolExecutor`, `pytest`, bash.

**Spec:** `src/tk26_vision/docs/superpowers/specs/2026-07-03-restaurant-resolution-and-waving-vlm-concurrency-design.md`

**Prior work reused as-is (no changes in this plan):** the HRI resolution-bump plan's shared depth-reprojection fix and `vision_driver.launch.py`'s `color_width`/`color_height` launch-arg passthrough.

## Global Constraints

- Restaurant color resolution target: **1920x1080**. Depth stays 640x576@30fps — untouched.
- Launch script: **`master_restaurant1.sh` only.** `master_restaurant2.sh` is explicitly out of scope — do not touch it, do not fix its unrelated typo, do not even read its uncommitted working-tree state as a reference.
- VLM waving-fallback measured latency: **3.9s-7.5s** across 5 live trials (720p and 1080p) — this is why "always launch concurrently" is viable; do not re-litigate this by assuming the old design's 5-20s ceiling.
- Early-exit threshold: **new declared ROS param `vlm_skip_min_wavers`, default `2`** — not hardcoded.
- Executor sizing: **`ThreadPoolExecutor(max_workers=2)`, not 1** — an abandoned call from an early-exited request can still be running when the next request comes in; 1 worker would silently reintroduce the wait.
- Termination semantics: **stop-waiting-and-discard only.** Do not implement true network-level cancellation (no `asyncio`/`AsyncOpenAI` migration) — this was explicitly evaluated and rejected in the spec.
- Blast radius: **Restaurant only.** `/detect_waving_persons`'s only real caller is `Restaurant/restaurants.py:255`. `Restaurant/restaurants_fake.py:122` (test/mock tree) and GPSR (separate, unrelated code path) must not be touched.
- `DetectWaving.srv`'s `min_waving_persons` field stays in the message, unused by the new logic — do not remove it, do not add a new `.srv` field.

---

### Task 1: Restaurant Orbbec resolution launch override

**Files:**
- Modify: `src/tk25_basic/src/scripts/master_restaurant1.sh`

**Interfaces:**
- Consumes: `color_width`/`color_height` launch args already added to `vision_driver.launch.py` by the prior HRI resolution-bump work (defaults `1280`/`720`).

This is the tk25_basic repo (separate git root from `tk26_vision`). Confirm current branch/state before committing — the HRI resolution-bump plan's Task 5 landed on branch `openpi-lock-venvs` per explicit user direction at the time; re-check `git branch --show-current` and `git status` in this repo before starting, since the branch/dirty-state situation may have changed since then, and do not assume it's still the same.

- [ ] **Step 1: Read the current live content of the target lines**

```bash
cd /home/tinker/tk25_ws/src/tk25_basic
grep -n "vision_driver.launch.py" src/scripts/master_restaurant1.sh
```

Expected: two matches, inside the `if [ -n "$DEV" ]` / `else` branches of the vision-window pane-0 setup (search for `tmux send-keys -t $SESSION:$WINDOW.0` immediately above each).

- [ ] **Step 2: Add the resolution override to both branches**

Change:

```bash
if [ -n "$DEV" ]; then
    tmux send-keys -t $SESSION:$WINDOW.0 \
        "export TINKER_VISION_SESSION_TS='${TINKER_VISION_SESSION_TS}' && source ~/tk25_ws/install/setup.zsh && ros2 launch vision_bringup vision_driver.launch.py device:=\"$DEV\" launch_robot_state_publisher:=false; exec zsh" C-m
else
    tmux send-keys -t $SESSION:$WINDOW.0 \
        "export TINKER_VISION_SESSION_TS='${TINKER_VISION_SESSION_TS}' && source ~/tk25_ws/install/setup.zsh && ros2 launch vision_bringup vision_driver.launch.py enable_pan_tilt:=false launch_robot_state_publisher:=false; exec zsh" C-m
fi
```

to:

```bash
if [ -n "$DEV" ]; then
    tmux send-keys -t $SESSION:$WINDOW.0 \
        "export TINKER_VISION_SESSION_TS='${TINKER_VISION_SESSION_TS}' && source ~/tk25_ws/install/setup.zsh && ros2 launch vision_bringup vision_driver.launch.py device:=\"$DEV\" launch_robot_state_publisher:=false color_width:=1920 color_height:=1080; exec zsh" C-m
else
    tmux send-keys -t $SESSION:$WINDOW.0 \
        "export TINKER_VISION_SESSION_TS='${TINKER_VISION_SESSION_TS}' && source ~/tk25_ws/install/setup.zsh && ros2 launch vision_bringup vision_driver.launch.py enable_pan_tilt:=false launch_robot_state_publisher:=false color_width:=1920 color_height:=1080; exec zsh" C-m
fi
```

If the live file's surrounding content doesn't match this exactly (e.g. it was edited since this plan was written), treat the live file as ground truth and apply the equivalent two-line addition (`color_width:=1920 color_height:=1080` appended to both `ros2 launch vision_bringup vision_driver.launch.py ...` invocations in the pane-0 block) rather than force-fitting this exact diff.

- [ ] **Step 3: Syntax-check and confirm both branches got the override**

```bash
bash -n src/scripts/master_restaurant1.sh
grep -c "color_width:=1920 color_height:=1080" src/scripts/master_restaurant1.sh
```

Expected: `bash -n` prints nothing (exit 0); `grep -c` prints `2`.

- [ ] **Step 4: Confirm no other file/line changed**

```bash
git diff --stat
```

Expected: exactly one file, `src/scripts/master_restaurant1.sh`, with 2 insertions / 2 deletions (or however many lines the actual live-file diff touched — but only this one file, and only the two target lines).

- [ ] **Step 5: Commit**

```bash
git add src/scripts/master_restaurant1.sh
git commit -m "$(cat <<'EOF'
feat(scripts): launch Restaurant's Orbbec driver at 1920x1080 instead of 720p

Same mechanism as the HRI resolution-bump work: vision_driver.launch.py's
color_width/color_height args already default to 1280/720 for every other
caller; only this task-specific script overrides them. master_restaurant2.sh
is explicitly untouched (unrelated, mid-migration, uncommitted script).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013Ei1BHsidgbudrkDS3yhuJ
EOF
)"
```

---

### Task 2: `should_wait_for_vlm` decision function

**Files:**
- Modify: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py`
- Modify: `src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py`

**Interfaces:**
- Produces: `should_wait_for_vlm(cv_waver_count: int, skip_min_wavers: int) -> bool` — pure function, no I/O, no node state. Returns `True` if the caller should block on the VLM future (CV hasn't found enough yet), `False` if it should abandon the future without waiting (CV already found `skip_min_wavers` or more). `skip_min_wavers <= 0` always returns `True` (never skip) — treats a non-positive threshold as "feature off," not "always skip immediately."

Task 3 imports and calls this function from `waving_person_server.py`; this task has no dependency on Task 3 and can run independently.

- [ ] **Step 1: Write the failing tests**

Append to `src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py`:

```python
from tk_vision_specialized._waving_vlm import should_wait_for_vlm  # noqa: E402


def test_should_wait_for_vlm_waits_below_threshold():
    assert should_wait_for_vlm(0, 2) is True
    assert should_wait_for_vlm(1, 2) is True


def test_should_wait_for_vlm_skips_at_or_above_threshold():
    assert should_wait_for_vlm(2, 2) is False
    assert should_wait_for_vlm(3, 2) is False


def test_should_wait_for_vlm_never_skips_when_threshold_non_positive():
    assert should_wait_for_vlm(5, 0) is True
    assert should_wait_for_vlm(5, -1) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
source .venv-vision-main/bin/activate
PYTHONPATH=src/tk_vision_specialized python3 -m pytest src/tk_vision_specialized/test/test_waving_vlm.py -v -k should_wait_for_vlm
```

Expected: `ImportError: cannot import name 'should_wait_for_vlm' from 'tk_vision_specialized._waving_vlm'`.

- [ ] **Step 3: Implement**

In `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py`, add this function near the other pure helpers (e.g. directly after `select_boxes`, before `_resolve_key`):

```python
def should_wait_for_vlm(cv_waver_count: int, skip_min_wavers: int) -> bool:
    """True if the caller should block on the VLM future; False if CV alone
    already found enough wavers and the VLM call should be abandoned (left
    running in the background, its result discarded when it lands).

    skip_min_wavers <= 0 means "never skip" (always wait) -- a 0 threshold
    would otherwise mean cv_waver_count >= 0 is trivially true, immediately
    discarding every call regardless of what CV found.
    """
    if skip_min_wavers <= 0:
        return True
    return cv_waver_count < skip_min_wavers
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src/tk_vision_specialized python3 -m pytest src/tk_vision_specialized/test/test_waving_vlm.py -v -k should_wait_for_vlm
```

Expected: 3 tests PASS.

- [ ] **Step 5: Run the package's full test suite (lint + existing tests)**

```bash
PYTHONPATH=src/tk_vision_specialized python3 -m pytest src/tk_vision_specialized/test/test_waving_vlm.py src/tk_vision_specialized/test/test_flake8.py src/tk_vision_specialized/test/test_pep257.py -v
```

Expected: all `test_waving_vlm.py` tests PASS. If `test_flake8`/`test_pep257` show pre-existing package-wide failures unrelated to this change (matching the pattern already documented in the HRI resolution-bump plan's Task 1/2/3 reports), confirm by checking that no failure references `_waving_vlm.py` or `test_waving_vlm.py` specifically — don't assume clean, verify.

- [ ] **Step 6: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py src/tk_vision_specialized/test/test_waving_vlm.py
git commit -m "$(cat <<'EOF'
feat(tk_vision_specialized): add should_wait_for_vlm decision helper

Pure function backing the concurrent-VLM-with-early-exit change to
waving_person_server.py (next task): given how many wavers CV already
found and a configurable skip threshold, decides whether to block on the
VLM future or abandon it. Kept import-light (no rclpy/mediapipe/ultralytics)
so it's testable without pulling in the heavy node module, matching this
file's existing pure-function convention (decode_box_xyxy, select_boxes).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013Ei1BHsidgbudrkDS3yhuJ
EOF
)"
```

---

### Task 3: Concurrent VLM call with early-exit in `waving_person_server.py`

**Files:**
- Modify: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py`

**Interfaces:**
- Consumes: `should_wait_for_vlm(cv_waver_count: int, skip_min_wavers: int) -> bool` (Task 2).
- Produces: `DetectWavingPersonsNode._start_vlm_call(self, rgb_image) -> Optional[Future]`, `DetectWavingPersonsNode._merge_vlm_result(self, vlm_result, points, validmask_points, header, request, person_records, waving_persons_centroids, waving_annotations, waving_masks, waving_sources) -> tuple[int, str]`, `DetectWavingPersonsNode._log_discarded_vlm_result(self, future: Future) -> None`. These replace `_vlm_augment` entirely — it is deleted, not kept alongside.

No new automated test for this task: `_start_vlm_call`/`_merge_vlm_result`/`_log_discarded_vlm_result` are ROS/threading glue around already-tested logic (`should_wait_for_vlm` from Task 2; `is_duplicate_box`/`centroid_from_box`/`request_waving_persons_chain` already covered by `test_waving_geometry.py`/`test_waving_vlm.py`). `DetectWavingPersonsNode.__init__` loads a real YOLO model and MediaPipe pose estimator, so instantiating it in a test is avoided in this codebase's existing convention (no `test_waving_person_server.py` exists today either). Verification is AST-parse + existing suite (catches import/syntax errors and unused-import lint regressions) plus the live verification in Task 4.

- [ ] **Step 1: Add the executor import**

In `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py`, add to the imports (after `import threading`, line 16):

```python
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
```

- [ ] **Step 2: Import `should_wait_for_vlm`**

Change:

```python
from ._waving_vlm import (
    request_waving_persons_chain,
    build_provider_models,
    has_provider_key,
    WavingVlmError,
)
```

to:

```python
from ._waving_vlm import (
    request_waving_persons_chain,
    build_provider_models,
    has_provider_key,
    should_wait_for_vlm,
    WavingVlmError,
)
```

- [ ] **Step 3: Add the `vlm_skip_min_wavers` param and the dedicated executor**

In `__init__`, change:

```python
        self.declare_parameter('vlm_dedup_iou', 0.3)
        self.enable_vlm_fallback = (
            self.get_parameter('enable_vlm_fallback').value)
        self.vlm_provider = self.get_parameter('vlm_provider').value
        self.vlm_fallback_provider = (
            self.get_parameter('vlm_fallback_provider').value)
        self.vlm_model_qwen = self.get_parameter('vlm_model_qwen').value
        self.vlm_model_gemini = self.get_parameter('vlm_model_gemini').value
        self.vlm_timeout_s = float(self.get_parameter('vlm_timeout_s').value)
        self.vlm_max_retries = int(self.get_parameter('vlm_max_retries').value)
        self.vlm_dedup_iou = float(self.get_parameter('vlm_dedup_iou').value)
        self._vlm_chain = self._resolve_provider_chain()
```

to:

```python
        self.declare_parameter('vlm_dedup_iou', 0.3)
        self.declare_parameter('vlm_skip_min_wavers', 2)
        self.enable_vlm_fallback = (
            self.get_parameter('enable_vlm_fallback').value)
        self.vlm_provider = self.get_parameter('vlm_provider').value
        self.vlm_fallback_provider = (
            self.get_parameter('vlm_fallback_provider').value)
        self.vlm_model_qwen = self.get_parameter('vlm_model_qwen').value
        self.vlm_model_gemini = self.get_parameter('vlm_model_gemini').value
        self.vlm_timeout_s = float(self.get_parameter('vlm_timeout_s').value)
        self.vlm_max_retries = int(self.get_parameter('vlm_max_retries').value)
        self.vlm_dedup_iou = float(self.get_parameter('vlm_dedup_iou').value)
        self.vlm_skip_min_wavers = int(
            self.get_parameter('vlm_skip_min_wavers').value)
        self._vlm_chain = self._resolve_provider_chain()
        # Dedicated pool for the concurrent VLM waving-fallback call. 2
        # workers, not 1: an abandoned call from an early-exited request (see
        # _start_vlm_call / detect_waving_callback) can still be finishing in
        # the background when the NEXT request submits its own call -- with
        # only 1 worker, that submission would queue behind the abandoned one
        # and silently reintroduce the wait this whole change exists to avoid.
        self._vlm_executor = ThreadPoolExecutor(max_workers=2)
```

- [ ] **Step 4: Shut down the executor in `destroy_node`**

Change:

```python
    def destroy_node(self):
        if self._viewer_proc is not None and self._viewer_proc.poll() is None:
            try:
                self._viewer_proc.terminate()
                self._viewer_proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                self._viewer_proc.kill()
        return super().destroy_node()
```

to:

```python
    def destroy_node(self):
        self._vlm_executor.shutdown(wait=False)
        if self._viewer_proc is not None and self._viewer_proc.poll() is None:
            try:
                self._viewer_proc.terminate()
                self._viewer_proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                self._viewer_proc.kill()
        return super().destroy_node()
```

- [ ] **Step 5: Replace `_vlm_augment` with `_start_vlm_call` + `_merge_vlm_result` + `_log_discarded_vlm_result`**

Delete the whole `_vlm_augment` method:

```python
    def _vlm_augment(self, rgb_image, points, validmask_points, header, request,
                     person_records, waving_persons_centroids,
                     waving_annotations, waving_masks, waving_sources):
        """Call the VLM chain and append the wavers MediaPipe missed.

        Mutates the four aligned waver lists in place. Returns
        (n_added, provider_used). Never raises: any VLM failure logs a warning
        and returns (0, '') so the service still answers with MediaPipe results.
        """
        try:
            result = request_waving_persons_chain(
                rgb_image, provider_models=self._vlm_chain,
                timeout_s=self.vlm_timeout_s, max_retries=self.vlm_max_retries,
                logger=self.get_logger())
        except WavingVlmError as exc:
            self.get_logger().warn(f'VLM waving fallback unavailable: {exc}')
            return 0, ''

        existing_boxes = [(a[0], a[1], a[2], a[3]) for a in waving_annotations]
        n_added = 0
        for box in result.boxes:
            if is_duplicate_box(box, existing_boxes,
                                iou_thresh=self.vlm_dedup_iou):
                continue
            out = centroid_from_box(points, validmask_points, box,
                                    person_records)
            if out is None:
                self.get_logger().info(
                    f'VLM box {box} skipped: no usable depth.')
                continue
            centroid, used_mask = out
            if (request.threshold_meters > 0
                    and centroid[2] > request.threshold_meters):
                self.get_logger().info(
                    f'VLM waver dropped: depth {centroid[2]:.2f}m > threshold '
                    f'{request.threshold_meters:.2f}m')
                continue
            point_stamped = PointStamped()
            point_stamped.header = header
            point_stamped.point.x = float(centroid[0])
            point_stamped.point.y = float(centroid[1])
            point_stamped.point.z = float(centroid[2])
            x1, y1, x2, y2 = box
            waving_persons_centroids.append(point_stamped)
            waving_annotations.append((x1, y1, x2, y2, None))
            waving_masks.append(used_mask)
            waving_sources.append('vlm')
            existing_boxes.append(box)
            n_added += 1
        return n_added, result.provider
```

Replace it with:

```python
    def _start_vlm_call(self, rgb_image):
        """Launch the VLM waving-fallback call on the dedicated executor.

        Returns None immediately if the fallback is disabled or no provider
        has a key configured -- callers treat None the same as "nothing to
        wait for, nothing to merge." Otherwise returns a Future whose
        .result() is a WavingVlmResult, or raises WavingVlmError on hard
        failure (matching request_waving_persons_chain's own contract).
        """
        if not self.enable_vlm_fallback or not self._vlm_chain:
            return None
        return self._vlm_executor.submit(
            request_waving_persons_chain, rgb_image,
            provider_models=self._vlm_chain,
            timeout_s=self.vlm_timeout_s, max_retries=self.vlm_max_retries,
            logger=self.get_logger(),
        )

    def _log_discarded_vlm_result(self, future: Future):
        """Done-callback for an abandoned (early-exited) VLM future.

        Never raises: swallows whatever the call eventually produced (result
        or exception) so an abandoned call finishing later doesn't surface as
        an unhandled-exception warning from the executor thread.
        """
        try:
            future.result()
            self.get_logger().debug(
                'Discarded VLM waving result (CV already found enough wavers).')
        except Exception as exc:  # noqa: BLE001 -- intentionally swallowed
            self.get_logger().debug(f'Discarded VLM waving call failed: {exc}')

    def _merge_vlm_result(self, vlm_result, points, validmask_points, header,
                          request, person_records, waving_persons_centroids,
                          waving_annotations, waving_masks, waving_sources):
        """Fold a completed WavingVlmResult into the CV-found waver lists.

        Mutates the four aligned waver lists in place. Returns
        (n_added, provider_used). Same dedup/centroid logic the old
        _vlm_augment had, just taking an already-computed result instead of
        fetching it itself.
        """
        existing_boxes = [(a[0], a[1], a[2], a[3]) for a in waving_annotations]
        n_added = 0
        for box in vlm_result.boxes:
            if is_duplicate_box(box, existing_boxes,
                                iou_thresh=self.vlm_dedup_iou):
                continue
            out = centroid_from_box(points, validmask_points, box,
                                    person_records)
            if out is None:
                self.get_logger().info(
                    f'VLM box {box} skipped: no usable depth.')
                continue
            centroid, used_mask = out
            if (request.threshold_meters > 0
                    and centroid[2] > request.threshold_meters):
                self.get_logger().info(
                    f'VLM waver dropped: depth {centroid[2]:.2f}m > threshold '
                    f'{request.threshold_meters:.2f}m')
                continue
            point_stamped = PointStamped()
            point_stamped.header = header
            point_stamped.point.x = float(centroid[0])
            point_stamped.point.y = float(centroid[1])
            point_stamped.point.z = float(centroid[2])
            x1, y1, x2, y2 = box
            waving_persons_centroids.append(point_stamped)
            waving_annotations.append((x1, y1, x2, y2, None))
            waving_masks.append(used_mask)
            waving_sources.append('vlm')
            existing_boxes.append(box)
            n_added += 1
        return n_added, vlm_result.provider
```

- [ ] **Step 6: Launch the VLM call as soon as the frame is copied**

In `detect_waving_callback`, change:

```python
            rgb_image = self.rgb_image.copy()
            depth_image = self.depth_image
            header = self.header
            camera_k = self.camera_k
        finally:
            self.img_lock.release()

        self.get_logger().info('Data copied for processing. Starting detection...')
```

to:

```python
            rgb_image = self.rgb_image.copy()
            depth_image = self.depth_image
            header = self.header
            camera_k = self.camera_k
        finally:
            self.img_lock.release()

        # Launch the VLM fallback now, in parallel with the depth conversion +
        # YOLO + MediaPipe pass below -- it only needs rgb_image, which is
        # already available. See _start_vlm_call / the merge-or-discard logic
        # after the CV loop.
        vlm_future = self._start_vlm_call(rgb_image)

        self.get_logger().info('Data copied for processing. Starting detection...')
```

- [ ] **Step 7: Replace the gated sequential call with wait-or-discard**

Change:

```python
        self.get_logger().info(f'Person candidates checked: {person_candidates}')

        n_vlm_added = 0
        vlm_provider_used = ''
        if (self._vlm_chain
                and request.min_waving_persons > 0
                and len(waving_persons_centroids) < request.min_waving_persons):
            n_vlm_added, vlm_provider_used = self._vlm_augment(
                rgb_image, points, validmask_points, header, request,
                person_records, waving_persons_centroids,
                waving_annotations, waving_masks, waving_sources,
            )
            self.get_logger().info(
                f'VLM fallback added {n_vlm_added} waver(s) '
                f'(provider={vlm_provider_used or "none"}).')
```

to:

```python
        self.get_logger().info(f'Person candidates checked: {person_candidates}')

        n_vlm_added = 0
        vlm_provider_used = ''
        if vlm_future is not None:
            if should_wait_for_vlm(
                    len(waving_persons_centroids), self.vlm_skip_min_wavers):
                try:
                    vlm_result = vlm_future.result(timeout=self.vlm_timeout_s)
                except (WavingVlmError, FutureTimeoutError) as exc:
                    self.get_logger().warn(
                        f'VLM waving fallback unavailable: {exc}')
                    vlm_result = None
                if vlm_result is not None:
                    n_vlm_added, vlm_provider_used = self._merge_vlm_result(
                        vlm_result, points, validmask_points, header, request,
                        person_records, waving_persons_centroids,
                        waving_annotations, waving_masks, waving_sources,
                    )
                    self.get_logger().info(
                        f'VLM fallback added {n_vlm_added} waver(s) '
                        f'(provider={vlm_provider_used or "none"}).')
            else:
                vlm_future.add_done_callback(self._log_discarded_vlm_result)
                self.get_logger().info(
                    f'CV already found {len(waving_persons_centroids)} '
                    f'waver(s) (>= vlm_skip_min_wavers='
                    f'{self.vlm_skip_min_wavers}); discarding VLM call '
                    f'without waiting.'
                )
```

- [ ] **Step 8: Verify the module parses cleanly and no import is left dangling**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
source .venv-vision-main/bin/activate
python3 -c "import ast; ast.parse(open('src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py').read())"
grep -n "_vlm_augment" src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py
```

Expected: AST parse succeeds silently; the `grep` for `_vlm_augment` returns **zero matches** (fully replaced, no stale call site left behind).

- [ ] **Step 9: Run the package's existing test suite**

```bash
PYTHONPATH=src/tk_vision_specialized python3 -m pytest src/tk_vision_specialized/test/ -v
```

Expected: `test_waving_vlm.py` (including Task 2's new tests) and `test_waving_geometry.py` PASS. If `test_flake8`/`test_pep257`/`test_copyright` show failures, verify none reference `waving_person_server.py` specifically before treating them as pre-existing/unrelated (same verification discipline as the HRI plan's Task 2/3 reports) — in particular check for an unused-import flake8 hit, since `WavingVlmError` is now used only in the `except` clause at the new call site (still needed) but confirm nothing else went stale.

- [ ] **Step 10: Commit**

```bash
git add src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py
git commit -m "$(cat <<'EOF'
feat(tk_vision_specialized): run waving VLM fallback concurrently with CV, early-exit at 2 wavers

_vlm_augment made the VLM chain call synchronously, gated behind
request.min_waving_persons -- a field Restaurant's BT node never actually
sets, so the fallback was dead code in production despite being enabled by
default. Splits it into _start_vlm_call (launches the VLM chain call on a
dedicated 2-worker ThreadPoolExecutor as soon as the synced frame is
available, concurrent with the CV/MediaPipe pass) and _merge_vlm_result
(the same dedup/centroid logic as before, now taking an already-resolved
result). detect_waving_callback blocks on the future only when CV found
fewer than vlm_skip_min_wavers (new param, default 2); otherwise the future
is abandoned via a swallowing done-callback. Not true cancellation -- the
abandoned call keeps running until it finishes or times out, its result
just goes unused; a real asyncio-based cancel was evaluated and rejected
as disproportionate. Restaurant is the only real caller of this service.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013Ei1BHsidgbudrkDS3yhuJ
EOF
)"
```

---

### Task 4: Live verification (operator-in-the-loop, requires hardware)

**Files:** None — no code changes. Gate before declaring this feature done.

No automated test can exercise the Orbbec hardware path or make live VLM API calls part of a CI-safe suite.

- [ ] **Step 1: Rebuild the touched packages**

```bash
cd /home/tinker/tk25_ws
./src/tk26_vision/scripts/build.sh --packages-select tk_vision_specialized vision_bringup
```

Expected: build succeeds with no errors.

- [ ] **Step 2: Launch Restaurant's driver and confirm 1080p + sustained rate**

```bash
bash ~/tk25_ws/src/tk25_basic/src/scripts/master_restaurant1.sh
```

In a separate shell:

```bash
source ~/tk25_ws/install/setup.zsh
ros2 topic echo /camera/color/camera_info --once | grep -E "width|height"
ros2 topic hz /camera/color/image_raw
```

Expected: `width: 1920`, `height: 1080`; `ros2 topic hz` reports ~30Hz sustained, no drop/SHM-overflow warnings in the Orbbec node's log.

- [ ] **Step 3: Verify the early-exit path (fast, no VLM wait)**

With 2 or more people waving in view of the Orbbec:

```bash
ros2 service call /detect_waving_persons tinker_vision_msgs_26/srv/DetectWaving "{threshold_meters: 5.0, target_frame: 'map', min_waving_persons: 0}"
```

Expected: response returns in well under ~2s. Check the node's log (the pane running `waving_person_server`, or `ros2 node info`/console output) for the line `discarding VLM call without waiting` — confirms the early-exit path fired and the VLM future was launched-then-abandoned, not awaited.

- [ ] **Step 4: Verify the wait/merge path (VLM engaged)**

With 0 or 1 people waving in view — ideally including one person far enough away or small enough in frame that MediaPipe is expected to miss them — call the same service. Expected: response takes several seconds (matching the measured 3.9-7.5s VLM latency, up to `vlm_timeout_s`=20s on a slow response). Check the log for `VLM fallback added N waver(s)` and inspect whether the far/small waver was recovered by the VLM path.

- [ ] **Step 5: Verify the fallback-disabled path is unaffected**

```bash
ros2 param set /detect_waving_persons_node enable_vlm_fallback false
```

Re-run the Step 4 scene. Expected: response stays fast (under ~2s) regardless of scene content — `_start_vlm_call` returns `None` immediately, matching pre-change behavior when the fallback is off. Reset the param back to `true` afterward if continuing to operate the robot.

- [ ] **Step 6: Record the outcome**

If all steps pass, this feature is complete. If Step 2's frame rate is degraded or shows SHM warnings, treat this the same as the HRI plan's Task 7 guidance — stop and re-examine the SHM segment sizing rather than assuming it's fine, since Restaurant's higher concurrent manip/nvblox load (noted in the design spec §3) makes this a real risk to check for, not a formality.
