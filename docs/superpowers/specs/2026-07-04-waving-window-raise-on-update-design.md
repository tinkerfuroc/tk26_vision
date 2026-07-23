# Waving Detection window: raise-to-front on each image update

**Date:** 2026-07-04
**Package:** `tk_vision_specialized`
**File:** `tk_vision_specialized/waving_person_server.py`
**Status:** approved, ready to implement

## Problem

The `detect_waving_persons` server can open a live cv2 popup window
(`show_window:=true`) titled **"Waving Detection"** that shows the annotated
debug frame for each service call. When the operator has other windows (a
terminal, rqt, RViz) covering it, a fresh detection frame paints *behind* those
windows and is easy to miss. The operator wants the window to surface to the
front whenever a new image is drawn.

## Goal

Each time the popup paints a **new** frame, raise the window to the front so the
latest detection is visible — then release the always-on-top constraint so the
operator can cover the window again between detections.

Explicitly **not** a permanent always-on-top window: raising happens only on an
actual image update, not continuously.

## Current architecture (unchanged by this change)

- A dedicated window thread `_cv2_window_loop` owns the single cv2 window for the
  life of the node (Qt5 highgui backend). GUI calls happen only on this thread.
- It creates the window with `cv2.namedWindow(..., WINDOW_NORMAL)` + a startup
  placeholder frame, then loops pulling annotated frames off `_frame_queue`
  (producer = the service callback, which drops stale frames and keeps the
  newest).
- Each frame is painted by the static helper `_show_frame(window_name, frame)`
  = `cv2.imshow` + 3× `cv2.waitKey(1)` pumps.
- On an empty queue the loop calls a bare `cv2.waitKey(1)` to keep the event
  loop responsive — this path does **not** call `_show_frame` and has no new
  image.

`_show_frame` therefore runs exactly once per new frame (plus once for the
startup placeholder), which makes it the single correct hook for
"each time the image updates."

## Design

Add a topmost toggle around the paint inside `_show_frame`:

1. `cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1.0)` — raises the
   window to the front.
2. `cv2.imshow(...)` + the existing 3× `cv2.waitKey(1)` — paint the frame and
   let Qt process the raise.
3. `cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0.0)` + one more
   `cv2.waitKey(1)` — release the always-on-top flag. The window remains
   frontmost where it was just raised, but the operator can cover it again until
   the next frame arrives.

### Robustness

`WND_PROP_TOPMOST` requires a Qt/GTK highgui backend. Both `setWindowProperty`
calls are wrapped in `try/except Exception` (swallowed) so a build without the
constant/backend still paints the frame — only the raise is skipped. This
mirrors the existing defensive `try/except` around `namedWindow` in
`_cv2_window_loop`. (Pre-verified: `cv2.WND_PROP_TOPMOST == 5` exists in the
venv's OpenCV 4.10.0 with the Qt5 backend.)

### Why this mechanism

OpenCV's highgui exposes no direct "raise window" call; toggling
`WND_PROP_TOPMOST` on → off is the OpenCV-native way to pull a window to the
front and then let it return to normal stacking. Toggling the Qt
stays-on-top hint may cause slight flicker on some window managers — accepted
tradeoff, and the reason permanent always-on-top was considered and rejected
(it would cover the terminal for the whole run).

## Scope / non-goals

- No change to the frame queue, the drop-stale-keep-newest producer, the
  `/detect_waving_debug_image` publish path, or the annotation drawing.
- No new ROS parameter — the behavior rides on the existing `show_window` flag.
- The startup placeholder also raises once (harmless: the window pops up when
  the node starts), which is acceptable.

## Verification

Live-GUI behavior; not headlessly unit-testable. Manual check on a host with an
X display:

1. Run `ros2 run tk_vision_specialized waving_person_server --ros-args -p show_window:=true`.
2. Cover the "Waving Detection" window with the terminal.
3. Call `detect_waving_persons`.
4. Confirm the window jumps to the front when the new frame paints, and that it
   can be covered again afterward.

Also confirm the file byte-compiles and (on a build host) `./scripts/build.sh
--packages-select tk_vision_specialized` succeeds.
