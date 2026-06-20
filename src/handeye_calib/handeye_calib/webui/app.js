// handeye_calib calibrate_web — v2 skeleton (T1).
// Vanilla JS, no build step. T2-T6 fill in the empty panel bodies; this file
// owns the WS connection, tab switching, connection pill, and the
// info-tab status lines that every later task reads from `state`.

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

// ---- global state shared with later tasks ----------------------------------
// `state` is the last WS message body. Tabs added in T2-T6 read from it.
let state = null;

// ---- status line helper ----------------------------------------------------
// Mirrors pan_tilt's status-line convention: `kind` is one of
// "", "ok", "warn", "err" — the class drives the colour via style.css.
function setStatus(id, text, kind = "") {
  const el = typeof id === "string" ? document.getElementById(id) : id;
  if (!el) return;
  el.textContent = text;
  el.className = "status-line" + (kind ? " " + kind : "");
}

// ---- tab switching ---------------------------------------------------------
function activateTab(name) {
  $$(".side-tab").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  $$(".side-panel-content").forEach(p => p.classList.toggle("active", p.dataset.panel === name));
}
$$(".side-tab").forEach(b => {
  b.addEventListener("click", () => activateTab(b.dataset.tab));
});

// ---- T2: resizable live camera panel --------------------------------------
// Camera floats top-left; controls wrap around it. A handle in the panel's
// bottom-right corner lets the operator drag to resize. Width clamped 240-800
// and persisted to localStorage under `handeye-cam-w` (the brief-named key).
const CAM_PANEL = $("#cam-panel");
const CAM_HANDLE = $("#cam-resize");
const CAM_IMG = $("#cam-img");
const CAM_KEY = "handeye-cam-w";
const CAM_MIN = 240, CAM_MAX = 800;

function clampCamWidth(px) {
  // Also bound by viewport so the panel can never swallow the whole window.
  const maxAllowed = Math.min(CAM_MAX, window.innerWidth - 80);
  return Math.max(CAM_MIN, Math.min(maxAllowed, px));
}
function setCamWidth(px) {
  const w = clampCamWidth(px);
  document.documentElement.style.setProperty("--cam-w", w + "px");
  return w;
}
(function initCamWidth() {
  const stored = parseInt(localStorage.getItem(CAM_KEY) || "", 10);
  setCamWidth(Number.isFinite(stored) ? stored : 480);
})();
if (CAM_HANDLE) {
  let dragging = false, startX = 0, startW = 0;
  CAM_HANDLE.addEventListener("pointerdown", (e) => {
    dragging = true;
    startX = e.clientX;
    startW = parseInt(getComputedStyle(document.documentElement)
      .getPropertyValue("--cam-w"), 10) || 480;
    CAM_HANDLE.classList.add("dragging");
    try { CAM_HANDLE.setPointerCapture(e.pointerId); } catch (_) {}
    e.preventDefault();
  });
  CAM_HANDLE.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    setCamWidth(startW + (e.clientX - startX));
  });
  const endDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    CAM_HANDLE.classList.remove("dragging");
    try { CAM_HANDLE.releasePointerCapture(e.pointerId); } catch (_) {}
    const cur = parseInt(getComputedStyle(document.documentElement)
      .getPropertyValue("--cam-w"), 10);
    if (Number.isFinite(cur)) localStorage.setItem(CAM_KEY, String(cur));
  };
  CAM_HANDLE.addEventListener("pointerup", endDrag);
  CAM_HANDLE.addEventListener("pointercancel", endDrag);
}
window.addEventListener("resize", () => {
  const cur = parseInt(getComputedStyle(document.documentElement)
    .getPropertyValue("--cam-w"), 10);
  if (Number.isFinite(cur)) setCamWidth(cur);
});

// ---- T2: live frame polling (3 Hz, cache-busted) --------------------------
// /api/frame.jpg returns the annotated overlay by default; `?raw=1` skips it.
// The `frame-mode` radio toggles between annotated and raw at the URL level
// so the server-side overlay budget controls bandwidth either way.
let frameMode = "annotated";

function currentFrameUrl() {
  const base = "/api/frame.jpg";
  const params = new URLSearchParams({ t: String(Date.now()) });
  if (frameMode === "raw") params.set("raw", "1");
  return `${base}?${params.toString()}`;
}
function refreshFrame() {
  if (CAM_IMG) CAM_IMG.src = currentFrameUrl();
}
// 3 Hz (~333 ms): matches the brief; smooth enough for board-positioning
// without saturating the wifi link to a remote operator station.
setInterval(refreshFrame, 333);
refreshFrame();  // kick off immediately so the panel isn't blank for a tick

// Placeholder visibility — driven by the <img>'s load/error events. The
// `cam-panel.no-frame` class swap is the CSS-only "show placeholder"
// mechanism mirrored from pan_tilt.
function markFrameMissing(missing) {
  if (CAM_PANEL) CAM_PANEL.classList.toggle("no-frame", missing);
}
if (CAM_IMG) {
  CAM_IMG.addEventListener("load", () => markFrameMissing(false));
  CAM_IMG.addEventListener("error", () => markFrameMissing(true));
}

// Mode radio handler — flip the URL on change so the next tick uses raw/ann.
$$('input[name="frame-mode"]').forEach((r) => {
  r.addEventListener("change", () => {
    const checked = document.querySelector('input[name="frame-mode"]:checked');
    frameMode = checked ? checked.value : "annotated";
    refreshFrame();  // pick up the new URL without waiting for the next tick
  });
});

// ---- WebSocket connection pill --------------------------------------------
const INDICATOR = $("#conn-indicator");

function setIndicator(text, kind) {
  if (!INDICATOR) return;
  INDICATOR.textContent = text;
  INDICATOR.classList.remove("connected", "dropped");
  if (kind) INDICATOR.classList.add(kind);
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(proto + "//" + location.host + "/ws");
  ws.onopen = () => setIndicator("WS: live", "connected");
  ws.onmessage = (ev) => {
    try { state = JSON.parse(ev.data); } catch (_) { return; }
    render();
  };
  const reconnect = () => {
    setIndicator("WS: dropped — retrying", "dropped");
    setTimeout(connectWS, 1500);
  };
  ws.onclose = reconnect;
  ws.onerror = () => { /* onclose fires after */ };
  return ws;
}
connectWS();

// ---- render the info tab from state ---------------------------------------
function render() {
  if (!state) return;

  if (state.camera_connected) {
    const hz = (state.frame_hz || 0).toFixed(1);
    const age = state.frame_age_sec === null || state.frame_age_sec === undefined
      ? "—"
      : (state.frame_age_sec < 10
          ? state.frame_age_sec.toFixed(2) + " s"
          : ">10 s (stale)");
    setStatus("info-camera", `camera: streaming ${hz} Hz · last frame ${age} ago`, "ok");
  } else {
    setStatus("info-camera",
      `camera: waiting on ${state.image_topic || "(no topic)"}`,
      "warn");
  }

  if (state.t_base_ee) {
    setStatus("info-tf", "tf: base→ee resolved", "ok");
  } else {
    setStatus("info-tf", "tf: waiting for base→ee", "warn");
  }

  // T2: detection badge in the lower-left of the camera panel. Reads
  // state.last_detection = {corners: int, reproj_px: float|null} (T1 schema)
  // and shows "corners=N rms=X.XXpx OK" (green) when the board is visible,
  // or "corners=0 NO DETECTION" (red) otherwise.
  const badge = document.getElementById("detection-badge");
  if (badge) {
    const ld = state.last_detection;
    if (ld && Number.isFinite(ld.corners) && ld.corners > 0) {
      const rms = Number.isFinite(ld.reproj_px) ? ld.reproj_px.toFixed(2) + "px" : "—";
      badge.textContent = `corners=${ld.corners}  rms=${rms}  OK`;
      badge.classList.remove("err");
      badge.classList.add("ok");
    } else {
      badge.textContent = "corners=0  NO DETECTION";
      badge.classList.remove("ok");
      badge.classList.add("err");
    }
  }

  // T2: keep the placeholder topic in sync with whatever image_topic the
  // server is currently subscribed to. Cheap idempotent assignment.
  const placeholderTopic = document.getElementById("placeholder-topic");
  if (placeholderTopic && state.image_topic) {
    placeholderTopic.textContent = state.image_topic;
  }
}
