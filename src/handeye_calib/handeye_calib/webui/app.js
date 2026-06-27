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

// ---- T3: Move tab — joint editor + presets --------------------------------
// Mirrors src/pan_tilt/webui/app.js's joint editor verbatim: 7 J0..J6 inputs,
// rad/deg radio toggle with `prevUnit` tracking to convert in place, Load
// current / Zero all helpers, and a confirm() dialog before any POST /api/move.
// Presets: Home = [0,0,0,0,0,0,0]; look-forward is wired as a button but
// emits a console.warn() until a verified safe joint set from a calibration
// session is recorded. The brief is literal: do NOT invent unsafe joint
// values — a button that warns is correct, a button that sends unsafe joints
// is not.
const N_JOINTS = 7;
const MOVE_INPUTS_EL = $("#move-joint-inputs");
const moveJointInputs = [];
if (MOVE_INPUTS_EL) {
  for (let i = 0; i < N_JOINTS; i++) {
    const row = document.createElement("div");
    row.className = "joint-row";
    row.innerHTML = `<label>J${i}</label><input type="number" step="0.01" value="0">`;
    MOVE_INPUTS_EL.appendChild(row);
    moveJointInputs.push(row.querySelector("input"));
  }
}

function moveUnit() {
  const checked = document.querySelector('input[name="move-unit"]:checked');
  return checked ? checked.value : "rad";
}
function readMoveJointsRad() {
  const unit = moveUnit();
  return moveJointInputs.map((inp) => {
    const v = parseFloat(inp.value) || 0;
    return unit === "deg" ? (v * Math.PI) / 180 : v;
  });
}
function writeMoveJoints(valuesRad) {
  const unit = moveUnit();
  moveJointInputs.forEach((inp, i) => {
    const v = valuesRad[i] !== undefined ? valuesRad[i] : 0;
    inp.value = (unit === "deg" ? (v * 180) / Math.PI : v).toFixed(4);
  });
}

// Unit-switch with prevUnit tracking so values stay numerically consistent
// across toggles (rad → deg → rad round-trips without drift). Copied from
// pan_tilt/webui/app.js to keep the two tools' UX identical.
let prevMoveUnit = "rad";
document.querySelectorAll('input[name="move-unit"]').forEach((r) => {
  r.addEventListener("change", () => {
    const nu = moveUnit();
    if (nu === prevMoveUnit) return;
    moveJointInputs.forEach((inp) => {
      const v = parseFloat(inp.value) || 0;
      const asRad = prevMoveUnit === "deg" ? (v * Math.PI) / 180 : v;
      inp.value = (nu === "deg" ? (asRad * 180) / Math.PI : asRad).toFixed(4);
    });
    prevMoveUnit = nu;
  });
});

const BTN_LOAD_CURRENT = $("#move-load-current");
if (BTN_LOAD_CURRENT) {
  BTN_LOAD_CURRENT.addEventListener("click", () => {
    if (!state || !state.xarm_joint_positions || state.xarm_joint_positions.length === 0) {
      alert("xArm joint_states not yet received");
      return;
    }
    writeMoveJoints(state.xarm_joint_positions.slice(0, N_JOINTS));
  });
}
const BTN_ZERO = $("#move-zero");
if (BTN_ZERO) {
  BTN_ZERO.addEventListener("click", () => writeMoveJoints(Array(N_JOINTS).fill(0)));
}

// applyMoveConfirm(angles_rad) — confirm + POST /api/move + status wiring.
// Exposed as a named function (per the brief's "Produces" list) so future
// tasks / smoke scripts can re-use the exact same client path. Status flow:
// warn ("moving…") → ok (server reason) or err (server reason / HTTP error).
async function applyMoveConfirm(anglesRad) {
  if (!Array.isArray(anglesRad) || anglesRad.length !== N_JOINTS) {
    setStatus("move-status", `expected ${N_JOINTS} joints, got ${anglesRad && anglesRad.length}`, "err");
    return;
  }
  if (!confirm("Send xArm to these joints now?\n" + anglesRad.map((a) => a.toFixed(4)).join(", "))) {
    return;
  }
  setStatus("move-status", "moving…", "warn");
  try {
    const r = await fetch("/api/move", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ joints: anglesRad }),
    });
    const body = await r.json();
    if (r.ok && body.ok) {
      setStatus("move-status", "move sent: " + (body.reason || "ok"), "ok");
    } else {
      setStatus("move-status", "FAIL: " + (body.reason || body.detail || ("HTTP " + r.status)), "err");
    }
  } catch (e) {
    setStatus("move-status", "ERROR: " + e, "err");
  }
}

const BTN_SEND = $("#move-send");
if (BTN_SEND) {
  BTN_SEND.addEventListener("click", () => applyMoveConfirm(readMoveJointsRad()));
}

// Presets: Home zeroes all seven joints (a verified safe rest pose on the
// xArm). look-forward intentionally warns until a calibration-session pose
// is recorded — see the TODO note in index.html.
const PRESET_BAR = $("#move-presets");
if (PRESET_BAR) {
  PRESET_BAR.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-preset]");
    if (!btn) return;
    const preset = btn.dataset.preset;
    if (preset === "home") {
      writeMoveJoints(Array(N_JOINTS).fill(0));
      setStatus("move-status", "preset loaded: home (zeros)", "");
    } else if (preset === "look-forward") {
      // TODO: look-forward preset pose not yet defined — pick a verified
      // safe joint set from a calibration session and wire here. Deliberately
      // NOT inventing joint values: the brief is literal about not shipping
      // an unsafe preset, so this button warns and does nothing else.
      console.warn(
        "TODO: look-forward preset pose not yet defined — pick a verified safe joint set from a calibration session and wire here"
      );
      setStatus(
        "move-status",
        "look-forward preset not wired yet (see TODO in app.js)",
        "warn"
      );
    }
  });
}

// ---- T3: Info tab — kv-tables + matrix + safety ---------------------------

// formatMatrix(rows, dp=4): joins a 2-D numeric array into the same monospace
// "+0.1234 -0.5678 …" rows the pan_tilt T_base_ee pre block uses, so the two
// tools' Info tabs render T_base_ee identically. Defaults to 4 dp per the
// brief.
function formatMatrix(rows, dp = 4) {
  if (!Array.isArray(rows) || rows.length === 0) return "–";
  return rows
    .map((row) =>
      row
        .map((v) => {
          const n = Number(v);
          if (!Number.isFinite(n)) return "  ?    ";
          return (n >= 0 ? " " : "") + n.toFixed(dp);
        })
        .join("  ")
    )
    .join("\n");
}

function kvSet(rootSel, key, value, kind = "") {
  const cell = document.querySelector(`${rootSel} td[data-k="${key}"]`);
  if (!cell) return;
  cell.textContent = value;
  cell.className = kind || "";
}

function renderInfoTab(s) {
  // ---- Camera ------------------------------------------------------------
  kvSet("#info-kv-camera", "image_topic", s.image_topic || "—");
  kvSet("#info-kv-camera", "ros_domain_id",
        s.ros_domain_id !== undefined && s.ros_domain_id !== null ? String(s.ros_domain_id) : "—");
  kvSet("#info-kv-camera", "status", s.camera_connected ? "streaming" : "—");
  kvSet("#info-kv-camera", "frame_count",
        s.frame_count !== undefined && s.frame_count !== null ? String(s.frame_count) : "0");
  if (s.frame_age_sec === null || s.frame_age_sec === undefined) {
    kvSet("#info-kv-camera", "frame_age", "—");
  } else {
    const age = Number(s.frame_age_sec);
    kvSet("#info-kv-camera", "frame_age", age < 10 ? age.toFixed(2) + " s" : ">10 s (stale)");
  }
  kvSet("#info-kv-camera", "frame_hz",
        Number.isFinite(s.frame_hz) && s.frame_hz > 0 ? s.frame_hz.toFixed(1) + " Hz" : "—");

  // ---- Robot state -------------------------------------------------------
  if (Array.isArray(s.xarm_joint_positions) && s.xarm_joint_positions.length > 0) {
    const joints = s.xarm_joint_positions.map((j) => Number(j).toFixed(4)).join(", ");
    kvSet("#info-kv-robot", "xarm_joints",
          `${s.xarm_joint_positions.length} joints: [${joints}]`);
  } else {
    kvSet("#info-kv-robot", "xarm_joints", "—");
  }
  kvSet("#info-kv-robot", "tf_status", s.t_base_ee ? "ok" : "waiting");

  // ---- T_base_eef matrix -------------------------------------------------
  const matEl = $("#info-matrix-tbe");
  if (matEl) {
    matEl.textContent = s.t_base_ee ? formatMatrix(s.t_base_ee, 4) : "–";
  }

  // ---- ChArUco board -----------------------------------------------------
  const b = s.board || {};
  const grid = (b.squares_x && b.squares_y) ? `${b.squares_x} × ${b.squares_y}` : "—";
  kvSet("#info-board", "grid", grid);
  kvSet("#info-board", "square_len",
        Number.isFinite(b.square_len_m) ? (b.square_len_m * 1000).toFixed(1) + " mm" : "—");
  kvSet("#info-board", "marker_len",
        Number.isFinite(b.marker_len_m) ? (b.marker_len_m * 1000).toFixed(1) + " mm" : "—");
  kvSet("#info-board", "aruco_dict", b.aruco_dict || "—");

  // ---- Safety envelope ---------------------------------------------------
  const safEl = $("#info-safety");
  if (safEl) {
    safEl.textContent = s.safety_envelope
      ? JSON.stringify(s.safety_envelope, null, 2)
      : "–";
  }
}

// renderMoveSafety: drive #move-safety-status off state.safety_preview, the
// server-side SafetyEnvelope verdict (added in T3 so the math doesn't have
// to be duplicated in JS). Shape: {safe: bool|null, detail: str}. Green
// when safe, red when violation, muted when the server can't decide (no TF
// / no envelope).
function renderMoveSafety(s) {
  const sp = s.safety_preview;
  if (!sp || sp.safe === null || sp.safe === undefined) {
    setStatus("move-safety-status",
              (sp && sp.detail) ? sp.detail : "safety: waiting for TF…",
              "");
    return;
  }
  setStatus("move-safety-status",
            sp.detail || (sp.safe ? "safe" : "VIOLATION"),
            sp.safe ? "ok" : "err");
}

// ---- T4: Capture tab — settle gate, gallery, diversity meter ---------------
// The capture button is HARD-gated by state.stability.steady (the v1 deferral
// the brief closes): the operator can't even fire the request until camera +
// intrinsics + detection + steady all line up. On accept the gallery refreshes
// from the next WS push; per-sample delete sends DELETE /api/samples/{idx} and
// also relies on the WS push to update the diversity meter + meta strip.
const CAPTURE_BTN = $("#capture-btn");
const CAPTURE_STAB = $("#capture-stability");

function _renderCaptureStability(s) {
  if (!CAPTURE_STAB) return;
  const stab = s.stability || {};
  const steady = !!stab.steady;
  const target = Number.isFinite(stab.target_frames) ? stab.target_frames : 0;
  const since = Number.isFinite(stab.since_frames) ? stab.since_frames : 0;
  CAPTURE_STAB.classList.remove("ok", "warn", "err");
  if (steady) {
    CAPTURE_STAB.textContent = `stability: steady ✓ (${since}/${target})`;
    CAPTURE_STAB.classList.add("ok");
  } else if (target > 0) {
    CAPTURE_STAB.textContent = `stability: stabilizing… ${since}/${target}`;
    CAPTURE_STAB.classList.add("warn");
  } else {
    CAPTURE_STAB.textContent = "stability: waiting for camera";
    CAPTURE_STAB.classList.add("err");
  }
}

function _captureReady(s) {
  // Mirror the server's HARD gate so the button never sends a doomed POST:
  // camera_connected && intrinsics_ok && last_detection.corners>0 && steady.
  if (!s) return false;
  if (!s.camera_connected) return false;
  if (!s.intrinsics_ok) return false;
  const ld = s.last_detection;
  if (!ld || !Number.isFinite(ld.corners) || ld.corners <= 0) return false;
  const stab = s.stability || {};
  return !!stab.steady;
}

if (CAPTURE_BTN) {
  CAPTURE_BTN.addEventListener("click", async () => {
    setStatus("capture-status", "capturing…", "warn");
    try {
      const r = await fetch("/api/capture", { method: "POST" });
      const body = await r.json();
      if (r.ok && body.ok) {
        setStatus("capture-status",
                  `accepted: ${body.reason || "ok"} (${body.num_samples} total)`,
                  "ok");
      } else {
        setStatus("capture-status",
                  `rejected: ${body.reason || ("HTTP " + r.status)}`,
                  "err");
      }
    } catch (e) {
      setStatus("capture-status", "ERROR: " + e, "err");
    }
  });
}

const btnAnchor = document.getElementById('btn-anchor');
if (btnAnchor) btnAnchor.onclick = async () => {
  const r = await fetch('/api/anchor', {method: 'POST'});
  const j = await r.json();
  setStatus('anchor-status', j.ok
    ? `head anchor: ${j.n_anchor_obs} obs (scatter ${j.scatter ? j.scatter.trans_mm.toFixed(1) : '?'}mm)`
    : `anchor failed: ${j.reason}`);
};
const btnAnchorClear = document.getElementById('btn-anchor-clear');
if (btnAnchorClear) btnAnchorClear.onclick = async () => {
  await fetch('/api/anchor/clear', {method: 'POST'});
  setStatus('anchor-status', 'no head anchor');
};

async function _deleteSample(idx) {
  if (!confirm(`Delete sample #${idx}?`)) return;
  setStatus("capture-status", `deleting #${idx}…`, "warn");
  try {
    const r = await fetch(`/api/samples/${idx}`, { method: "DELETE" });
    const body = await r.json();
    if (r.ok && body.ok) {
      setStatus("capture-status",
                `deleted #${idx} (${body.num_samples} remaining)`,
                "ok");
    } else {
      setStatus("capture-status",
                `delete failed: ${body.reason || ("HTTP " + r.status)}`,
                "err");
    }
  } catch (e) {
    setStatus("capture-status", "ERROR: " + e, "err");
  }
}

// Same click-eat bug class as renderWaypointsList: every render() destroys
// and rebuilds the gallery's per-row ✕ delete button. Cache a samples
// signature so we only rebuild when state.samples actually changes.
let _lastGallerySig = null;
function _renderGallery(s) {
  const gal = document.getElementById("gallery");
  if (!gal) return;
  const samples = Array.isArray(s.samples) ? s.samples : [];
  const sig = JSON.stringify(samples);
  if (sig === _lastGallerySig) return;
  _lastGallerySig = sig;
  if (samples.length === 0) {
    gal.innerHTML = '<div class="gallery-empty" id="gallery-empty">no samples captured yet</div>';
    return;
  }
  // Build new rows, then swap in-place so the operator's scroll position
  // and any in-flight image loads survive a WS push.
  const frag = document.createDocumentFragment();
  for (const m of samples) {
    const idx = Number(m.idx);
    const item = document.createElement("div");
    item.className = "gallery-item";
    const thumb = document.createElement("img");
    thumb.className = "gallery-thumb";
    thumb.alt = `sample ${idx}`;
    thumb.loading = "lazy";
    thumb.src = `/api/samples/${idx}/thumb.jpg`;
    thumb.addEventListener("error", () => { thumb.style.visibility = "hidden"; });
    item.appendChild(thumb);

    const meta = document.createElement("div");
    meta.className = "gallery-meta";
    const corners = Number.isFinite(m.n_corners) ? m.n_corners : "?";
    const rms = Number.isFinite(m.reproj_px) ? m.reproj_px.toFixed(2) + "px" : "—";
    const area = Number.isFinite(m.area_frac)
      ? (m.area_frac * 100).toFixed(1) + "%"
      : "—";
    const ang = Number.isFinite(m.angular_delta_deg)
      ? "Δ" + m.angular_delta_deg.toFixed(1) + "°"
      : "Δ—";
    meta.innerHTML =
      `<span class="gallery-idx">#${idx}</span> · corners ${corners}<br>` +
      `rms ${rms} · area ${area} · ${ang}`;
    item.appendChild(meta);

    const del = document.createElement("button");
    del.type = "button";
    del.className = "gallery-delete";
    del.title = `delete sample ${idx}`;
    del.textContent = "✕";
    del.addEventListener("click", () => _deleteSample(idx));
    item.appendChild(del);

    frag.appendChild(item);
  }
  gal.innerHTML = "";
  gal.appendChild(frag);
}

function _renderDiversityMeter(s) {
  const fill = document.getElementById("diversity-fill");
  const label = document.getElementById("diversity-label");
  if (!fill || !label) return;
  const d = s.diversity || {};
  const cov = Number.isFinite(d.coverage_deg) ? d.coverage_deg : 0;
  const target = Number.isFinite(d.target_deg) && d.target_deg > 0
    ? d.target_deg
    : 30;
  const pct = Math.max(0, Math.min(100, (cov / target) * 100));
  fill.style.width = pct + "%";
  fill.classList.remove("ok", "warn");
  if (pct >= 100) fill.classList.add("ok");
  else if (pct >= 50) fill.classList.add("warn");
  label.textContent = `${cov.toFixed(1)}° / ${target.toFixed(0)}°`;
}

function renderCaptureTab(s) {
  _renderCaptureStability(s);
  if (CAPTURE_BTN) CAPTURE_BTN.disabled = !_captureReady(s);
  _renderDiversityMeter(s);
  _renderGallery(s);
}

// ---- T4: Auto-capture sequence UI — run / dry-run / cancel + live progress -
// Three buttons + a progress text line + a scrollable log.  State comes from
// state.sequence (pushed by T3 CaptureSequenceRunner via WS):
//   {running, dry_run, current_idx, total, current_step, log: list[str]}
const SEQ_RUN    = $("#sequence-run-btn");
const SEQ_DRY    = $("#sequence-dry-btn");
const SEQ_CANCEL = $("#sequence-cancel-btn");
const SEQ_PROG   = $("#sequence-progress");
const SEQ_LOG    = $("#sequence-log");

async function startSequence(dryRun) {
  const total = (state.waypoints || []).length;
  const verb = dryRun ? "dry-run (move + settle only)" : "RUN CAPTURE";
  if (!confirm(`${verb} the ${total}-waypoint sequence?\nThe arm will move to each pose in order. Click Cancel to stop.`)) return;
  setStatus("sequence-status", "starting…", "warn");
  const r = await fetch("/api/sequence/start", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({dry_run: dryRun})});
  const body = await r.json();
  if (!body.ok) setStatus("sequence-status", `start failed: ${body.reason}`, "err");
}

if (SEQ_RUN)    SEQ_RUN.addEventListener("click", () => startSequence(false));
if (SEQ_DRY)    SEQ_DRY.addEventListener("click", () => startSequence(true));
if (SEQ_CANCEL) SEQ_CANCEL.addEventListener("click", async () => {
  setStatus("sequence-status", "cancelling…", "warn");
  await fetch("/api/sequence/cancel", {method: "POST"});
});

// SEQ_LOG has no clickable items so it doesn't eat clicks, but rewriting
// its innerHTML on every 10 Hz push is a needless CPU/bandwidth burn —
// guard the log rewrite on a signature so it only fires when the bounded
// runner-log actually changed.
let _lastSeqLogSig = null;
function renderSequenceUI() {
  if (!SEQ_RUN) return;
  const seq = (state && state.sequence) || {running: false, current_step: "idle", total: 0, log: []};
  const wps = (state && state.waypoints) || [];
  const canRun = wps.length > 0 && !seq.running;
  SEQ_RUN.disabled    = !canRun;
  SEQ_DRY.disabled    = !canRun;
  SEQ_CANCEL.disabled = !seq.running;
  if (seq.running) {
    SEQ_PROG.textContent = `${seq.current_step} — #${seq.current_idx ?? "?"} / ${seq.total}`;
    SEQ_PROG.className = "status-line warn";
  } else {
    SEQ_PROG.textContent = seq.current_step === "done" ? "done" :
                           seq.current_step === "cancelled" ? "cancelled" : "idle";
    SEQ_PROG.className = "status-line " + (seq.current_step === "done" ? "ok" :
                                             seq.current_step === "cancelled" ? "warn" : "");
  }
  const logSig = JSON.stringify(seq.log);
  if (logSig !== _lastSeqLogSig) {
    _lastSeqLogSig = logSig;
    SEQ_LOG.innerHTML = seq.log.map(line => `<li>${line.replace(/</g, "&lt;")}</li>`).join("");
  }
}

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

  // T3: Info-tab kv-tables / matrix / board / safety + Move-tab safety line.
  renderInfoTab(state);
  renderSettings(state);
  renderMoveSafety(state);
  // T4: Capture-tab stability badge + gallery + diversity meter.
  renderCaptureTab(state);
  // T2: Waypoints sub-panel (above manual capture button).
  renderWaypointsList();
  // T4: Auto-capture sequence UI (below waypoints, above manual capture).
  renderSequenceUI();
}

// ---- Calibration settings (Info tab): calib_frame + depth knobs + emitter ----
// Mirrors state.config from the WS push into the form, and POSTs /api/config on
// Apply. While the operator is mid-edit (configDirty), the 10 Hz render loop must
// NOT clobber their inputs — so we only sync from state when not dirty.
let configDirty = false;
let emitterDirty = false;  // separate: the emitter command is ONLY sent when the
                           // operator actually touched the checkbox, so a routine
                           // depth/frame Apply in color mode never kills the projector.
const CFG_INPUTS = [
  "use-ffs-depth-input", "depth-weight-input",
  "depth-sigma-input", "depth-win-input", "depth-min-corners-input",
];
function markConfigDirty() { configDirty = true; }
CFG_INPUTS.forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", markConfigDirty);
});
$$('input[name="calib-frame"]').forEach((r) => r.addEventListener("change", markConfigDirty));
const EMITTER_INPUT = document.getElementById("ir-emitter-input");
if (EMITTER_INPUT) EMITTER_INPUT.addEventListener("change", () => { configDirty = true; emitterDirty = true; });

function renderSettings(s) {
  const cfg = s.config;
  if (!cfg || configDirty) return;  // don't clobber in-progress edits
  const r = document.querySelector(`input[name="calib-frame"][value="${cfg.calib_frame}"]`);
  if (r) r.checked = true;
  const set = (id, v) => { const el = document.getElementById(id); if (el && document.activeElement !== el) el.value = v; };
  const chk = (id, v) => { const el = document.getElementById(id); if (el && document.activeElement !== el) el.checked = !!v; };
  chk("use-ffs-depth-input", cfg.use_ffs_depth);
  if (cfg.ir_emitter_enabled !== null && cfg.ir_emitter_enabled !== undefined) {
    chk("ir-emitter-input", cfg.ir_emitter_enabled);
  }
  set("depth-weight-input", cfg.depth_weight);
  set("depth-sigma-input", cfg.depth_sigma_m);
  set("depth-win-input", cfg.depth_win);
  set("depth-min-corners-input", cfg.depth_min_corners);
}

async function applyConfig() {
  const frame = document.querySelector('input[name="calib-frame"]:checked')?.value || "color";
  const curFrame = state && state.config ? state.config.calib_frame : "color";
  const n = state ? (state.num_samples || 0) : 0;
  if (frame !== curFrame && n > 0 &&
      !confirm(`Switching to the ${frame} frame discards ${n} captured sample(s) (they're tied to the current frame's intrinsics). Continue?`)) {
    return;
  }
  const body = {
    calib_frame: frame,
    use_ffs_depth: document.getElementById("use-ffs-depth-input")?.checked,
    depth_weight: parseFloat(document.getElementById("depth-weight-input")?.value),
    depth_sigma_m: parseFloat(document.getElementById("depth-sigma-input")?.value),
    depth_win: parseInt(document.getElementById("depth-win-input")?.value, 10),
    depth_min_corners: parseInt(document.getElementById("depth-min-corners-input")?.value, 10),
  };
  // Only command the IR emitter when the operator actually toggled it — a depth/
  // frame tweak must NOT silently disable the projector (needed for color depth).
  if (emitterDirty) body.ir_emitter_enabled = document.getElementById("ir-emitter-input")?.checked;
  setStatus("config-status", "applying…", "warn");
  try {
    const resp = await fetch("/api/config", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await resp.json();
    if (resp.ok && j.ok) {
      configDirty = false;  // success -> let the WS state re-sync the form
      emitterDirty = false;
      let msg = `applied (frame=${j.calib_frame || frame})`;
      if (j.emitter) msg += j.emitter.ok ? "; emitter set" : `; emitter FAILED: ${j.emitter.reason}`;
      setStatus("config-status", msg, j.emitter && !j.emitter.ok ? "warn" : "ok");
    } else {
      // Keep dirty so a failed apply doesn't let the WS clobber unsaved edits.
      setStatus("config-status", "FAIL: " + (j.reason || ("HTTP " + resp.status)), "err");
    }
  } catch (e) {
    setStatus("config-status", "ERROR: " + e, "err");
  }
}
const APPLY_CFG_BTN = document.getElementById("apply-config-btn");
if (APPLY_CFG_BTN) APPLY_CFG_BTN.addEventListener("click", applyConfig);

// ---- T2: Waypoints sub-panel — list + add-current + delete + save/reload --
// Lives in the Capture tab, above the manual capture button. Reads
// state.waypoints (array of {idx, abbrev, joints_rad}) on every WS push.
// Load fills the Move-tab joint inputs via writeMoveJoints(); Delete fires
// DELETE /api/waypoints/{idx}; Save/Reload hit the matching POST endpoints.
// All confirm()/status semantics copied verbatim from pan_tilt/webui/app.js.
const WP_LIST     = document.getElementById("waypoints-list");
const WP_ADD_BTN  = document.getElementById("waypoint-add-current-btn");
const WP_SAVE_BTN = document.getElementById("waypoint-save-btn");
const WP_REL_BTN  = document.getElementById("waypoint-reload-btn");

if (WP_ADD_BTN) WP_ADD_BTN.addEventListener("click", async () => {
  setStatus("waypoints-status", "adding…", "warn");
  try {
    const r = await fetch("/api/waypoints", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
    const body = await r.json();
    setStatus("waypoints-status",
      body.ok ? `added — ${body.count} waypoint(s)` : `add failed: ${body.reason}`,
      body.ok ? "ok" : "err");
  } catch (e) {
    setStatus("waypoints-status", "ERROR: " + e, "err");
  }
});

if (WP_SAVE_BTN) WP_SAVE_BTN.addEventListener("click", async () => {
  if (!confirm("Save the current waypoint sequence to disk?\n(Existing per-robot waypoints YAML will be backed up first.)")) return;
  setStatus("waypoints-status", "saving…", "warn");
  try {
    const r = await fetch("/api/waypoints/save", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
    const body = await r.json();
    setStatus("waypoints-status",
      body.ok ? `saved to ${body.path}` : `save failed: ${body.reason}`,
      body.ok ? "ok" : "err");
  } catch (e) {
    setStatus("waypoints-status", "ERROR: " + e, "err");
  }
});

if (WP_REL_BTN) WP_REL_BTN.addEventListener("click", async () => {
  setStatus("waypoints-status", "reloading…", "warn");
  try {
    const r = await fetch("/api/waypoints/reload", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
    const body = await r.json();
    setStatus("waypoints-status",
      body.ok ? `loaded ${body.count} waypoint(s) from ${body.path}` : `reload failed: ${body.reason}`,
      body.ok ? "ok" : "err");
  } catch (e) {
    setStatus("waypoints-status", "ERROR: " + e, "err");
  }
});

// Cache the last-rendered waypoint signature so we only rewrite WP_LIST.innerHTML
// when state.waypoints ACTUALLY changes. Without this guard, render() (called
// from every 10 Hz WS push) destroys + recreates the list DOM 10 times/sec
// unconditionally; per-row button clicks landing in the destroy/rebuild window
// get suppressed by the browser (mousedown on old button, mouseup on new DOM
// node = no click event fires), forcing the operator to "click multiple times".
// pan_tilt's calib_web only re-renders its waypoint list after user actions for
// the same reason. (See systematic-debugging session 2026-06-20.)
let _lastWaypointsSig = null;
function renderWaypointsList() {
  if (!WP_LIST) return;
  const wps = (state && Array.isArray(state.waypoints)) ? state.waypoints : [];
  // Signature must include every field the render reads: idx + joints_rad
  // (for the tooltip + Load) + abbrev (for the visible label). Stringify is
  // ~1µs for typical N<30 — cheap relative to the innerHTML rewrite it saves.
  const sig = JSON.stringify(wps);
  if (sig === _lastWaypointsSig) return;
  _lastWaypointsSig = sig;
  if (wps.length === 0) {
    WP_LIST.innerHTML = '<li class="waypoints-empty">no waypoints recorded yet</li>';
    return;
  }
  WP_LIST.innerHTML = wps.map(w =>
    `<li class="waypoint-row" data-idx="${w.idx}">
       <span class="waypoint-idx">#${w.idx}</span>
       <span class="waypoint-joints" title="${w.joints_rad.map(j => j.toFixed(4)).join(', ')} rad">${w.abbrev}</span>
       <span class="waypoint-actions">
         <button data-act="load" type="button">Load</button>
         <button data-act="delete" type="button">Delete</button>
       </span>
     </li>`
  ).join("");
}

// Delegate clicks for per-row buttons (Load fills the move-tab joint inputs;
// Delete fires DELETE /api/waypoints/{idx}).
if (WP_LIST) {
  WP_LIST.addEventListener("click", async (ev) => {
    const btn = ev.target.closest("button[data-act]");
    if (!btn) return;
    const row = btn.closest("li[data-idx]");
    if (!row) return;
    const idx = parseInt(row.dataset.idx, 10);
    if (btn.dataset.act === "load") {
      const wp = state && Array.isArray(state.waypoints) && state.waypoints.find(w => w.idx === idx);
      if (wp) {
        writeMoveJoints(wp.joints_rad);
        setStatus("waypoints-status", `loaded #${idx} into Move tab`, "");
      }
    } else if (btn.dataset.act === "delete") {
      if (!confirm(`Delete waypoint #${idx}?`)) return;
      try {
        const r = await fetch(`/api/waypoints/${idx}`, {method: "DELETE"});
        const body = await r.json();
        setStatus("waypoints-status",
          body.ok ? `deleted #${idx} — ${body.count} remaining` : `delete failed: ${body.reason}`,
          body.ok ? "ok" : "err");
      } catch (e) {
        setStatus("waypoints-status", "ERROR: " + e, "err");
      }
    }
  });
}

// ---- T5: Solve tab — method picker, comparison table, canvases -----------
// One Solve button drives POST /api/solve {method: ...}. The server returns
// solve_payload_v2 (mm/deg/px pre-rendered) on success or {ok:false,reason}
// on a degraded path (no samples / no intrinsics / etc). All three canvases
// are vanilla canvas2d — no chart library, no SVG. The brief is literal:
// histogram = simple bar chart, scatter = dots at (i, v[i]), coverage =
// project T_cam_board * [0,0,0] to image with K.
const SOLVE_BTN = $("#solve-btn");
const SOLVE_METHOD = $("#solve-method");
const SOLVE_STATUS = "solve-status";
const SOLVE_VERDICT = $("#solve-verdict");
const METHOD_TABLE = $("#method-table");
let lastSolve = null;  // last successful solve payload — drives the coverage canvas + re-renders

// Canvas helpers --------------------------------------------------------------
function _getCtx(canvasId) {
  const c = typeof canvasId === "string" ? document.getElementById(canvasId) : canvasId;
  if (!c || !c.getContext) return null;
  return c.getContext("2d");
}

function drawHistogram(canvasId, values, opts = {}) {
  // Simple bar chart with N bins (default 20). Values < 0 are clamped to 0.
  const ctx = _getCtx(canvasId);
  if (!ctx) return;
  const c = ctx.canvas;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.strokeStyle = "#2c3140";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, c.width - 1, c.height - 1);
  if (!Array.isArray(values) || values.length === 0) {
    ctx.fillStyle = "#8a93a6";
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText("no data", 8, 16);
    return;
  }
  const bins = Math.max(2, Math.min(60, opts.bins || 20));
  const vmin = 0;
  const vmax = Math.max(1e-6, ...values.map((v) => Math.max(0, Number(v) || 0)));
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const x = Math.max(vmin, Math.min(vmax, Number(v) || 0));
    let idx = Math.floor(((x - vmin) / (vmax - vmin)) * bins);
    if (idx >= bins) idx = bins - 1;
    counts[idx]++;
  }
  const cmax = Math.max(...counts) || 1;
  const pad = 18;
  const w = c.width - 2 * pad;
  const h = c.height - 2 * pad;
  const bw = w / bins;
  ctx.fillStyle = "#4da3ff";
  for (let i = 0; i < bins; i++) {
    const bh = (counts[i] / cmax) * h;
    ctx.fillRect(pad + i * bw + 0.5, pad + (h - bh), Math.max(1, bw - 1), bh);
  }
  ctx.fillStyle = "#8a93a6";
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillText("0", pad, c.height - 4);
  ctx.fillText(vmax.toFixed(2) + " px", c.width - pad - 40, c.height - 4);
  ctx.fillText("N=" + values.length, pad, 12);
}

function drawScatter(canvasId, values, opts = {}) {
  // Dots at (i, v[i]). x = sample index, y = residual px. Useful for spotting
  // a single bad sample dominating the RMS.
  const ctx = _getCtx(canvasId);
  if (!ctx) return;
  const c = ctx.canvas;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.strokeStyle = "#2c3140";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, c.width - 1, c.height - 1);
  if (!Array.isArray(values) || values.length === 0) {
    ctx.fillStyle = "#8a93a6";
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText("no data", 8, 16);
    return;
  }
  const n = values.length;
  const vmax = Math.max(1e-6, ...values.map((v) => Math.max(0, Number(v) || 0)));
  const pad = 18;
  const w = c.width - 2 * pad;
  const h = c.height - 2 * pad;
  ctx.fillStyle = "#5fd37f";
  for (let i = 0; i < n; i++) {
    const x = pad + (n === 1 ? w / 2 : (i / (n - 1)) * w);
    const v = Math.max(0, Number(values[i]) || 0);
    const y = pad + (h - (v / vmax) * h);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#8a93a6";
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillText("0", pad, c.height - 4);
  ctx.fillText("i=" + (n - 1), c.width - pad - 40, c.height - 4);
  ctx.fillText("max=" + vmax.toFixed(2) + "px", pad, 12);
}

function _projectBoardCentroid(sample, K) {
  // For each sample, project the board origin (T_cam_board * [0,0,0,1]) to
  // pixel coordinates via the K-pinhole. Returns {x, y, z} where z is the
  // depth in metres (used for warm-vs-cool colouring).
  // sample is expected to expose `T_cam_board` as a 4x4 row-major list-of-
  // lists; we tolerate flat 16-element arrays too. K is 3x3 (list-of-lists or
  // flat 9-element). If anything is missing we return null and the caller
  // skips the dot.
  if (!sample || !K) return null;
  const T = sample.T_cam_board || sample.t_cam_board;
  if (!T) return null;
  const fx = K[0] && K[0][0] !== undefined ? K[0][0] : (Array.isArray(K) ? K[0] : null);
  const fy = K[1] && K[1][1] !== undefined ? K[1][1] : (Array.isArray(K) ? K[4] : null);
  const cx = K[0] && K[0][2] !== undefined ? K[0][2] : (Array.isArray(K) ? K[2] : null);
  const cy = K[1] && K[1][2] !== undefined ? K[1][2] : (Array.isArray(K) ? K[5] : null);
  if ([fx, fy, cx, cy].some((v) => v === null || v === undefined)) return null;
  // Board origin in camera frame == T_cam_board[:3, 3]
  let tx, ty, tz;
  if (Array.isArray(T[0])) {
    tx = T[0][3]; ty = T[1][3]; tz = T[2][3];
  } else {
    tx = T[3]; ty = T[7]; tz = T[11];
  }
  if (!Number.isFinite(tz) || tz <= 0) return null;
  return { x: fx * (tx / tz) + cx, y: fy * (ty / tz) + cy, z: tz };
}

function drawCoverage(canvasId, samples, K, opts = {}) {
  // Plot the projected board centroid for each sample, colored by depth
  // (warm = closer). Assumes a typical RealSense 640x480 image grid; if the
  // K matrix's principal point implies a larger image, we expand the visible
  // box to cover it.
  const ctx = _getCtx(canvasId);
  if (!ctx) return;
  const c = ctx.canvas;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.strokeStyle = "#2c3140";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, c.width - 1, c.height - 1);
  if (!Array.isArray(samples) || samples.length === 0 || !K) {
    ctx.fillStyle = "#8a93a6";
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText("no samples", 8, 16);
    return;
  }
  // Heuristic image bounds from the K principal point.
  let imgW = 640, imgH = 480;
  try {
    const cx = Array.isArray(K[0]) ? K[0][2] : K[2];
    const cy = Array.isArray(K[1]) ? K[1][2] : K[5];
    if (Number.isFinite(cx)) imgW = Math.max(imgW, Math.round(2 * cx));
    if (Number.isFinite(cy)) imgH = Math.max(imgH, Math.round(2 * cy));
  } catch (_) {}
  const pad = 6;
  const sx = (c.width - 2 * pad) / imgW;
  const sy = (c.height - 2 * pad) / imgH;
  const s = Math.min(sx, sy);
  const offX = (c.width - imgW * s) / 2;
  const offY = (c.height - imgH * s) / 2;
  // Draw image-bound rectangle so the operator can see the camera frame.
  ctx.strokeStyle = "#3a4255";
  ctx.strokeRect(offX, offY, imgW * s, imgH * s);

  const projected = samples.map((s_) => _projectBoardCentroid(s_, K)).filter(Boolean);
  if (projected.length === 0) {
    ctx.fillStyle = "#8a93a6";
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText("no projected centroids (missing T_cam_board?)", 8, 16);
    return;
  }
  const zmin = Math.min(...projected.map((p) => p.z));
  const zmax = Math.max(...projected.map((p) => p.z));
  const zr = Math.max(1e-6, zmax - zmin);
  ctx.font = "10px JetBrains Mono, monospace";
  projected.forEach((p, i) => {
    const t = (p.z - zmin) / zr;
    // Warm (closer, t=0) -> orange/red; cool (farther, t=1) -> blue.
    const r = Math.round(255 * (1 - t));
    const g = Math.round(120 + 80 * (1 - t));
    const b = Math.round(80 + 175 * t);
    const px = offX + p.x * s;
    const py = offY + p.y * s;
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#e3e7ef";
    ctx.fillText(String(i), px + 5, py - 5);
  });
  // Depth legend
  ctx.fillStyle = "#8a93a6";
  ctx.fillText(
    `N=${projected.length}  z=[${zmin.toFixed(2)}, ${zmax.toFixed(2)}] m`,
    8, c.height - 6);
}

function _setKv(rootSel, key, text) {
  const cell = document.querySelector(`${rootSel} td[data-k="${key}"]`);
  if (cell) cell.textContent = text;
}

function _renderVerdict(status) {
  if (!SOLVE_VERDICT) return;
  SOLVE_VERDICT.hidden = !status;
  SOLVE_VERDICT.className = "gate-pill";
  if (!status) return;
  const cls = String(status).toLowerCase();
  if (cls === "pass" || cls === "warn" || cls === "fail") {
    SOLVE_VERDICT.classList.add(cls);
  }
  SOLVE_VERDICT.textContent = String(status);
}

function _renderMethodTable(rows) {
  if (!METHOD_TABLE) return;
  METHOD_TABLE.innerHTML = "";
  const thead = document.createElement("thead");
  thead.innerHTML = "<tr><th>method</th><th>reproj (px)</th></tr>";
  METHOD_TABLE.appendChild(thead);
  const tbody = document.createElement("tbody");
  if (!Array.isArray(rows) || rows.length === 0) {
    const tr = document.createElement("tr");
    tr.className = "empty";
    tr.innerHTML = '<td colspan="2">(no methods reported)</td>';
    tbody.appendChild(tr);
  } else {
    const bestPx = Math.min(...rows.map((r) => Number(r.reproj_px)));
    for (const r of rows) {
      const tr = document.createElement("tr");
      const px = Number(r.reproj_px);
      if (Number.isFinite(px) && Math.abs(px - bestPx) < 1e-9) tr.className = "best";
      tr.innerHTML = `<td>${r.name}</td><td class="num">${Number.isFinite(px) ? px.toFixed(3) : "?"}</td>`;
      tbody.appendChild(tr);
    }
  }
  METHOD_TABLE.appendChild(tbody);
}

function renderSolvePayload(p) {
  lastSolve = p;
  _renderVerdict(p.status);
  _renderMethodTable(p.per_method_summary || []);
  const tm = p.train_metrics_mm_deg || {};
  const hm = p.heldout_metrics_mm_deg || {};
  _setKv("#solve-metrics", "train_trans",
    Number.isFinite(tm.trans_rmse_mm) ? tm.trans_rmse_mm.toFixed(3) + " mm" : "–");
  _setKv("#solve-metrics", "train_rot",
    Number.isFinite(tm.rot_rmse_deg) ? tm.rot_rmse_deg.toFixed(3) + " deg" : "–");
  _setKv("#solve-metrics", "train_reproj",
    Number.isFinite(tm.reproj_px) ? tm.reproj_px.toFixed(3) + " px" : "–");
  _setKv("#solve-metrics", "held_trans",
    Number.isFinite(hm.trans_rmse_mm) ? hm.trans_rmse_mm.toFixed(3) + " mm" : "–");
  _setKv("#solve-metrics", "held_rot",
    Number.isFinite(hm.rot_rmse_deg) ? hm.rot_rmse_deg.toFixed(3) + " deg" : "–");
  _setKv("#solve-metrics", "held_reproj",
    Number.isFinite(hm.reproj_px) ? hm.reproj_px.toFixed(3) + " px" : "–");
  const xyz = p.X_xyz_mm || [];
  const rpy = p.X_rpy_deg || [];
  _setKv("#solve-X", "X_xyz_mm",
    xyz.length === 3 ? `[${xyz.map((v) => Number(v).toFixed(3)).join(", ")}] mm` : "–");
  _setKv("#solve-X", "X_rpy_deg",
    rpy.length === 3 ? `[${rpy.map((v) => Number(v).toFixed(3)).join(", ")}] deg` : "–");

  const resids = p.per_sample_reproj_px || [];
  drawHistogram("resid-hist", resids, { bins: 20 });
  drawScatter("resid-scatter", resids);
  // Coverage uses sample metadata (T_cam_board + K) which T4 will publish via
  // state.samples + state.K. Until then we fall back to whatever the latest
  // state push exposes; an empty array just yields the "no samples" message.
  const samples = (state && Array.isArray(state.samples)) ? state.samples : [];
  const K = (state && state.K) || (state && state.intrinsics && state.intrinsics.K) || null;
  drawCoverage("coverage", samples, K);
  if (p.observability && p.observability.ok === false) {
    setStatus(SOLVE_STATUS,
      'WARN: ' + p.observability.detail);
  }
}

async function runSolve() {
  if (!SOLVE_METHOD) return;
  const method = SOLVE_METHOD.value || "auto";
  setStatus(SOLVE_STATUS, "solving…", "warn");
  if (SOLVE_BTN) SOLVE_BTN.disabled = true;
  try {
    const r = await fetch("/api/solve", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ method }),
    });
    const body = await r.json();
    if (!r.ok || !body.ok) {
      _renderVerdict(null);
      setStatus(SOLVE_STATUS,
        "FAIL: " + (body.reason || body.detail || ("HTTP " + r.status)),
        "err");
      return;
    }
    const verdict = body.status || "PASS";
    const kind = verdict === "PASS" ? "ok" : (verdict === "WARN" ? "warn" : "err");
    setStatus(SOLVE_STATUS, `solve ${verdict} (method=${method})`, kind);
    renderSolvePayload(body);
  } catch (e) {
    _renderVerdict(null);
    setStatus(SOLVE_STATUS, "ERROR: " + e, "err");
  } finally {
    if (SOLVE_BTN) SOLVE_BTN.disabled = false;
  }
}

if (SOLVE_BTN) SOLVE_BTN.addEventListener("click", runSolve);

// Expose the canvas helpers + solve handler on `window` so manual smoke tests
// and the brief's "Produces" list both have an inspectable surface.
window.drawHistogram = drawHistogram;
window.drawScatter = drawScatter;
window.drawCoverage = drawCoverage;
window.runSolve = runSolve;
window.renderSolvePayload = renderSolvePayload;

// ---- T6: Promote tab — yaml + xacro unified-diff preview + apply ---------
// Two parallel renderers, one per half. Each Show button GETs
// /api/promote/diff once and paints the appropriate half; the cached diff
// is reused across both halves' Show clicks so the operator can flip back
// and forth without re-fetching. Each Apply button confirm()s, POSTs
// /api/promote/apply {which: 'yaml'|'xacro'}, then renders the backup-path
// (server returns it as `body.<half>.backup_path`). Reload-from-disk hits
// /api/promote/reload which clears last_solve so the operator can re-solve.
//
// Per the brief: the xacro half is ROBOT_NAME-scoped — when the server
// returns `body.xacro === null` the half shows a yellow "ROBOT_NAME unset"
// banner and its Apply button stays disabled. `mode === 'refuse-vendor'`
// shows a red banner (vendor path refusal) and likewise locks Apply.
const PROMOTE_YAML_DIFF_BTN  = $("#promote-yaml-diff-btn");
const PROMOTE_YAML_APPLY_BTN = $("#promote-yaml-apply-btn");
const PROMOTE_YAML_DIFF_PRE  = $("#promote-yaml-diff");
const PROMOTE_YAML_TARGET    = $("#yaml-target");
const PROMOTE_YAML_MODE      = $("#yaml-mode");
const PROMOTE_YAML_STATUS    = "promote-yaml-status";
const PROMOTE_YAML_BACKUP    = $("#promote-yaml-backup");

const PROMOTE_XACRO_DIFF_BTN  = $("#promote-xacro-diff-btn");
const PROMOTE_XACRO_APPLY_BTN = $("#promote-xacro-apply-btn");
const PROMOTE_XACRO_DIFF_PRE  = $("#promote-xacro-diff");
const PROMOTE_XACRO_TARGET    = $("#xacro-target");
const PROMOTE_XACRO_MODE      = $("#xacro-mode");
const PROMOTE_XACRO_WARN      = $("#xacro-warn");
const PROMOTE_XACRO_STATUS    = "promote-xacro-status";
const PROMOTE_XACRO_BACKUP    = $("#promote-xacro-backup");

const PROMOTE_RELOAD_BTN     = $("#promote-reload-btn");
const PROMOTE_RELOAD_STATUS  = "promote-reload-status";

let lastPromoteDiff = null;  // last GET /api/promote/diff body

function _renderDiffPre(preEl, diffText) {
  // Cheap unified-diff highlighter: split lines, classify by prefix.
  // No syntax-highlighter dependency; just three colour classes for
  // additions / deletions / hunk headers. Lines starting with '---' / '+++'
  // (the unified-diff file markers) are also classed as hunk-ish.
  if (!preEl) return;
  preEl.innerHTML = "";
  if (!diffText) {
    preEl.textContent = "(no changes — proposed matches current on disk)";
    return;
  }
  const lines = diffText.split("\n");
  const frag = document.createDocumentFragment();
  for (const line of lines) {
    const span = document.createElement("span");
    if (line.startsWith("+++") || line.startsWith("---")) {
      span.className = "diff-hunk";
    } else if (line.startsWith("@@")) {
      span.className = "diff-hunk";
    } else if (line.startsWith("+")) {
      span.className = "diff-add";
    } else if (line.startsWith("-")) {
      span.className = "diff-del";
    }
    span.textContent = line + "\n";
    frag.appendChild(span);
  }
  preEl.appendChild(frag);
}

function _applyPromoteHalfUI(half, halfBody) {
  // half ∈ {"yaml", "xacro"}; halfBody is one of body.yaml / body.xacro from
  // GET /api/promote/diff (may be null when ROBOT_NAME is unset for the xacro
  // half). Populates the badge, mode line, warn banner, diff pre, and the
  // Apply button's disabled state for the named half.
  const isYaml = half === "yaml";
  const targetEl = isYaml ? PROMOTE_YAML_TARGET : PROMOTE_XACRO_TARGET;
  const modeEl   = isYaml ? PROMOTE_YAML_MODE   : PROMOTE_XACRO_MODE;
  const warnEl   = isYaml ? null                : PROMOTE_XACRO_WARN;
  const preEl    = isYaml ? PROMOTE_YAML_DIFF_PRE : PROMOTE_XACRO_DIFF_PRE;
  const applyBtn = isYaml ? PROMOTE_YAML_APPLY_BTN : PROMOTE_XACRO_APPLY_BTN;

  if (warnEl) warnEl.hidden = true;
  if (halfBody === null || halfBody === undefined) {
    // ROBOT_NAME unset (xacro half) — yaml is always populated when
    // ok=true, so this branch is xacro-only in practice.
    if (targetEl) targetEl.textContent = "(unresolved)";
    if (modeEl)   modeEl.textContent = "ROBOT_NAME unset — yaml-only promote";
    if (preEl)    preEl.innerHTML = "";
    if (applyBtn) applyBtn.disabled = true;
    return;
  }
  if (targetEl) targetEl.textContent = halfBody.target_path || "(no path)";
  const mode = halfBody.mode || "?";
  const modeLabel =
    mode === "patch" ? "mode: patch (existing file)"
    : mode === "seed" ? "mode: seed (new per-robot xacro)"
    : mode === "refuse-vendor" ? "mode: refused (shared vendor xacro)"
    : `mode: ${mode}`;
  if (modeEl) modeEl.textContent = modeLabel;
  if (warnEl && halfBody.warning) {
    warnEl.hidden = false;
    warnEl.textContent = halfBody.warning;
  }
  _renderDiffPre(preEl, halfBody.diff || "");
  // Enable Apply unless the diff is empty (nothing to write) or this is a
  // vendor-path refusal. An empty diff is a no-op; the operator shouldn't
  // even bother clicking, and the server would write an identical file
  // with a fresh backup for no reason.
  const hasChanges = !!(halfBody.diff && halfBody.diff.length > 0);
  if (applyBtn) {
    applyBtn.disabled = !(hasChanges && mode !== "refuse-vendor");
  }
}

async function fetchPromoteDiff(quiet = false) {
  if (!quiet) setStatus(PROMOTE_YAML_STATUS, "loading diff…", "warn");
  try {
    const r = await fetch("/api/promote/diff");
    const body = await r.json();
    if (!r.ok || !body.ok) {
      setStatus(PROMOTE_YAML_STATUS,
        "FAIL: " + (body.reason || ("HTTP " + r.status)), "err");
      setStatus(PROMOTE_XACRO_STATUS,
        "FAIL: " + (body.reason || ("HTTP " + r.status)), "err");
      return null;
    }
    lastPromoteDiff = body;
    _applyPromoteHalfUI("yaml", body.yaml);
    _applyPromoteHalfUI("xacro", body.xacro);
    if (!quiet) {
      const robot = body.robot_name || "(ROBOT_NAME unset)";
      setStatus(PROMOTE_YAML_STATUS, `diff loaded · robot=${robot}`, "ok");
      setStatus(PROMOTE_XACRO_STATUS,
        body.xacro === null
          ? "ROBOT_NAME unset — yaml-only promote"
          : `diff loaded · robot=${robot}`,
        body.xacro === null ? "warn" : "ok");
    }
    return body;
  } catch (e) {
    setStatus(PROMOTE_YAML_STATUS, "ERROR: " + e, "err");
    setStatus(PROMOTE_XACRO_STATUS, "ERROR: " + e, "err");
    return null;
  }
}

async function applyPromote(which) {
  const diff = lastPromoteDiff || await fetchPromoteDiff(true);
  if (!diff) return;
  const half = which === "yaml" ? diff.yaml : diff.xacro;
  if (!half || !half.target_path) {
    const statusId = which === "yaml" ? PROMOTE_YAML_STATUS : PROMOTE_XACRO_STATUS;
    setStatus(statusId, "no target path resolved", "err");
    return;
  }
  if (!confirm(
    `Overwrite ${half.target_path}?\n` +
    `A timestamped backup will be made.`)) {
    return;
  }
  const statusId = which === "yaml" ? PROMOTE_YAML_STATUS : PROMOTE_XACRO_STATUS;
  const backupEl = which === "yaml" ? PROMOTE_YAML_BACKUP : PROMOTE_XACRO_BACKUP;
  setStatus(statusId, "writing…", "warn");
  try {
    const r = await fetch("/api/promote/apply", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ which }),
    });
    const body = await r.json();
    // Per-half result lives at body[which]; on a single-half failure the
    // top-level body.ok may be false too (the server surfaces single-half
    // failures at top-level for clarity).
    const halfRes = body && body[which];
    const ok = halfRes && halfRes.ok !== false && halfRes.written_path;
    if (ok) {
      setStatus(statusId, `wrote ${halfRes.written_path}`, "ok");
      if (backupEl) {
        backupEl.className = "status-line ok";
        backupEl.textContent = halfRes.backup_path
          ? `backup: ${halfRes.backup_path}`
          : "(no prior file — no backup made)";
      }
      // Re-fetch the diff so the next Apply round sees the empty-diff state
      // (and the Apply button auto-disables for a no-op).
      await fetchPromoteDiff(true);
    } else {
      const reason = (halfRes && halfRes.reason)
        || body.reason
        || ("HTTP " + r.status);
      setStatus(statusId, "FAIL: " + reason, "err");
      if (backupEl) {
        backupEl.className = "status-line";
        backupEl.textContent = "";
      }
    }
  } catch (e) {
    setStatus(statusId, "ERROR: " + e, "err");
  }
}

async function reloadPromote() {
  setStatus(PROMOTE_RELOAD_STATUS, "reloading…", "warn");
  try {
    const r = await fetch("/api/promote/reload", { method: "POST" });
    const body = await r.json();
    if (r.ok && body.ok) {
      setStatus(PROMOTE_RELOAD_STATUS,
        "reload: " + (body.reason || "ok") + " — run solve again",
        "ok");
      lastPromoteDiff = null;
      // Clear the per-half UI so stale diffs don't linger.
      _applyPromoteHalfUI("yaml", null);
      _applyPromoteHalfUI("xacro", null);
      if (PROMOTE_YAML_DIFF_PRE)  PROMOTE_YAML_DIFF_PRE.innerHTML = "";
      if (PROMOTE_XACRO_DIFF_PRE) PROMOTE_XACRO_DIFF_PRE.innerHTML = "";
      if (PROMOTE_YAML_BACKUP)    { PROMOTE_YAML_BACKUP.textContent = ""; PROMOTE_YAML_BACKUP.className = "status-line"; }
      if (PROMOTE_XACRO_BACKUP)   { PROMOTE_XACRO_BACKUP.textContent = ""; PROMOTE_XACRO_BACKUP.className = "status-line"; }
      setStatus(PROMOTE_YAML_STATUS, "");
      setStatus(PROMOTE_XACRO_STATUS, "");
    } else {
      setStatus(PROMOTE_RELOAD_STATUS,
        "FAIL: " + (body.reason || body.detail || ("HTTP " + r.status)),
        "err");
    }
  } catch (e) {
    setStatus(PROMOTE_RELOAD_STATUS, "ERROR: " + e, "err");
  }
}

if (PROMOTE_YAML_DIFF_BTN)   PROMOTE_YAML_DIFF_BTN.addEventListener("click", () => fetchPromoteDiff());
if (PROMOTE_XACRO_DIFF_BTN)  PROMOTE_XACRO_DIFF_BTN.addEventListener("click", () => fetchPromoteDiff());
if (PROMOTE_YAML_APPLY_BTN)  PROMOTE_YAML_APPLY_BTN.addEventListener("click", () => applyPromote("yaml"));
if (PROMOTE_XACRO_APPLY_BTN) PROMOTE_XACRO_APPLY_BTN.addEventListener("click", () => applyPromote("xacro"));
if (PROMOTE_RELOAD_BTN)      PROMOTE_RELOAD_BTN.addEventListener("click", reloadPromote);

window.fetchPromoteDiff = fetchPromoteDiff;
window.applyPromote = applyPromote;
window.reloadPromote = reloadPromote;
