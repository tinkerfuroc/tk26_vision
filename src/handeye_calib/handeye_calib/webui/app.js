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
}
