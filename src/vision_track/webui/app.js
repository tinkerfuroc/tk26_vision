/* track_web dashboard client: WS state feed, MJPEG video, click-to-reseed. */
"use strict";

const REACQ = {0: ["TRACKING", "tracking"], 1: ["PASSIVE", "passive"],
               2: ["NEEDS HELP", "needs-help"]};
const $ = (id) => document.getElementById(id);
let lastState = null;
let lastStateAt = 0;
let searching = false;      // fsm_state === 'initializing' (init/search phase)
let searchStartedTs = null; // node wall-clock (s) anchor for the elapsed timer
let waveBoxes = [];
let lastMode = "—";   // bench | observer | idle (from the /api/status poll)
let recording = false;  // rosbag record state (from the /api/status poll)

function log(msg) {
  const li = document.createElement("li");
  li.innerHTML = `<b>${new Date().toLocaleTimeString()}</b> ${msg}`;
  $("log").prepend(li);
  while ($("log").children.length > 80) $("log").lastChild.remove();
}

function renderState(s) {
  const prev = lastState;
  lastState = s;
  lastStateAt = Date.now();
  const badge = $("reacq-badge");
  if (s.fsm_state === "idle" || s.fsm_state === "initializing") {
    badge.textContent = s.fsm_state.toUpperCase();
    badge.className = "reacq";                       // neutral gray
  } else {
    const [label, cls] = REACQ[s.reacquisition_state] || ["?", ""];
    badge.textContent = label;
    badge.className = "reacq " + cls;
  }
  // Init liveness: during the (≈10 s, pre-lock) search the FSM is
  // 'initializing' and every state field is "—", which is indistinguishable
  // from a dead dashboard. Show an animated "Searching for target…" banner
  // with an elapsed-seconds timer (anchored to the node's search_started_ts;
  // animates without a count if the ts is absent). renderSearching() also runs
  // on a 1 Hz interval so the timer advances between ws pushes.
  searching = s.fsm_state === "initializing";
  searchStartedTs = (searching && s.search_started_ts != null)
    ? s.search_started_ts : null;
  renderSearching();
  $("fsm").textContent = s.fsm_state ?? "—";
  $("lost").textContent = s.target_lost;
  $("ids").textContent = `${s.target_track_id ?? "—"} (orig ${s.original_track_id ?? "—"})`;
  $("frames").textContent = s.frames_lost;
  $("hold").textContent = s.awaiting_help
    ? (s.active_help_timeout_sec > 0
        ? `${Math.max(0, s.active_help_timeout_sec - s.time_since_seen).toFixed(1)}s`
        : "∞ (waving)")
    : "—";
  const f = (x) => (x == null ? "—" : x.toFixed(3));
  $("sims").textContent = `${f(s.best_sim)} / ${f(s.second_sim)}`;
  if (prev) {
    if (prev.target_lost !== s.target_lost)
      log(s.target_lost ? "target LOST" : "target reacquired");
    if (prev.reacquisition_state !== s.reacquisition_state)
      log(`reacq → ${(REACQ[s.reacquisition_state] || ["?"])[0]}`);
  }
}

/* Init "Searching for target…" banner. Driven by renderState (on/off + ts)
   and a 1 Hz tick (timer advance). search_started_ts is the node's wall-clock,
   so the elapsed value is robust to client/server clock skew only at the few-
   hundred-ms level — fine for a coarse seconds counter; if it ever reads
   negative we clamp to 0. Absent ts ⇒ spinner only, no count. */
function renderSearching() {
  const box = $("searching");
  if (!box) return;
  box.classList.toggle("hidden", !searching);
  if (!searching) return;
  const txt = $("searching-text");
  if (searchStartedTs != null) {
    const elapsed = Math.max(0, Date.now() / 1000 - searchStartedTs);
    txt.textContent = `Searching for target… ${elapsed.toFixed(0)}s`;
  } else {
    txt.textContent = "Searching for target…";
  }
}
setInterval(renderSearching, 1000);

function renderGallery(g) {
  $("gal-meta").textContent = `v${g.version} · ${g.thumbs.length} views`;
  const div = $("gallery");
  div.innerHTML = "";
  g.thumbs.forEach((b64, i) => {
    if (!b64) return;
    const img = document.createElement("img");
    img.src = "data:image/png;base64," + b64;
    if (i === 0) img.classList.add("anchor");
    img.title = i === 0 ? "anchor view" : `view ${i}`;
    div.appendChild(img);
  });
}

/* Follow-mode bringup: one radio selects the group (follow_vision = vision+audio
   tracking with no base motion, follow_nav = audio + follow_server + nav BT),
   and a single Start/Stop pair posts that group to the ProcessManager. */
function selectedFollowMode() {
  const r = document.querySelector('input[name="follow-mode"]:checked');
  return r ? r.value : 'follow_vision';
}

function setFollowControlsRunning(running) {
  document.querySelectorAll('input[name="follow-mode"]').forEach((el) => {
    el.disabled = running;
  });
  $("follow-start").disabled = running;
  $("follow-stop").disabled = !running;
}

/* follow_server FollowState enum mirrored from the JSON status topic. */
const FOLLOW_STATES = {0: 'IDLE', 1: 'TRACKING', 2: 'PURSUIT_LAST_SEEN',
                       3: 'APPROACHING_FINAL', 4: 'SUCCEEDED', 5: 'FAILED'};

function renderFollow(f) {
  const el = $("follow-state");
  if (!el) return;
  // Stale (no /follow_server/status in ~2 s, e.g. vision-only mode or Nav2
  // down) or an empty payload -> the panel reads "—".
  if (!f || f.stale || f.state === undefined) {
    el.textContent = 'Follow state: —';
    return;
  }
  const name = FOLLOW_STATES[f.state] ?? `state ${f.state}`;
  const d = (f.distance_to_person >= 0)
    ? `, ${f.distance_to_person.toFixed(2)} m` : '';
  el.textContent = `Follow state: ${name}${d}${f.goal_held ? ' (HOLDING)' : ''}`;
}

function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws/state`);
  ws.onopen = () => { $("conn").textContent = "live"; $("conn").className = "badge on"; };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") renderState(msg.data);
    if (msg.type === "gallery") renderGallery(msg.data);
    if (msg.type === "follow") renderFollow(msg.data);
  };
  ws.onclose = () => {
    $("conn").textContent = "reconnecting…";
    $("conn").className = "badge off";
    setTimeout(connectWS, 1500);
  };
}

/* Map a click on the displayed <img> to native pixel coords. */
function clickToNative(ev) {
  const img = $("video");
  const r = img.getBoundingClientRect();
  if (!img.naturalWidth) return null;
  return [(ev.clientX - r.left) * img.naturalWidth / r.width,
          (ev.clientY - r.top) * img.naturalHeight / r.height];
}

async function post(url, body) {
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: body ? {"Content-Type": "application/json"} : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    return await r.json();
  } catch (e) { return {message: `request failed: ${e}`}; }
}

async function reseed(bbox, label) {
  const r = await post("/api/reseed", {bbox: bbox.map(Math.round)});
  if (r.success && lastMode === "idle") {
    // The tracker accepted the re-lock but no goal is active anywhere —
    // typically the active-help hold expired and the goal aborted (gallery
    // reset) while the operator was mid-interaction.
    log(`reseed(${label}) → accepted (id=${r.target_track_id}) but NO ACTIVE ` +
        `GOAL — did the hold expire? Start a goal and retry.`);
  } else {
    log(`reseed(${label}) → ${r.success ? "OK id=" + r.target_track_id : "FAIL"} (${r.message})`);
  }
  clearOverlays();
}

$("video").addEventListener("click", (ev) => {
  const pt = clickToNative(ev);
  if (!pt || !lastState) return;
  const hits = (lastState.candidates || []).filter((c) => c.bbox &&
    pt[0] >= c.bbox[0] && pt[0] <= c.bbox[2] &&
    pt[1] >= c.bbox[1] && pt[1] <= c.bbox[3]);
  if (!hits.length) { log("click: no candidate box there"); return; }
  hits.sort((a, b) => (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1])
                    - (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]));
  reseed(hits[0].bbox, `candidate ${hits[0].id}`);
});

function clearOverlays() { waveBoxes = []; $("overlays").innerHTML = ""; }

function renderWaveBoxes() {
  const img = $("video");
  const ov = $("overlays");
  ov.innerHTML = "";
  if (!img.naturalWidth) return;
  const r = img.getBoundingClientRect();
  const sx = r.width / img.naturalWidth, sy = r.height / img.naturalHeight;
  waveBoxes.forEach((b, i) => {
    const d = document.createElement("div");
    d.className = "wave-box";
    d.style.left = b[0] * sx + "px";
    d.style.top = b[1] * sy + "px";
    d.style.width = (b[2] - b[0]) * sx + "px";
    d.style.height = (b[3] - b[1]) * sy + "px";
    d.title = `waving person ${i} — click to re-seed`;
    d.onclick = (e) => { e.stopPropagation(); reseed(b, `wave ${i}`); };
    ov.appendChild(d);
  });
}

$("btn-start").onclick = async () => log("start → " + (await post("/api/goal/start")).message);
$("btn-stop").onclick = async () => log("stop → " + (await post("/api/goal/stop")).message);
$("btn-clear").onclick = clearOverlays;
$("btn-wave").onclick = async () => {
  const button = $("btn-wave");
  button.disabled = true;
  try {
    log("DetectWaving…");
    const r = await post("/api/wave");
    // waving action payload: 0 = wavers found, 1 = ran but none waving,
    // -1 (or transport error) = genuine failure.
    if (r.error || r.status === -1) { log(`wave FAIL (${r.error || "status " + r.status})`); return; }
    if (r.status === 1 || !r.boxes.length) { log("wave → no wavers detected"); return; }
    if (r.auto_reseeded !== undefined) {
      // single waver → auto-reseed (wave-to-resume), no click needed
      const rs = r.reseed || {};
      log(`wave → single waver, auto-reseed ${r.auto_reseeded
          ? "OK id=" + rs.target_track_id : "FAILED (" + (rs.message || "?") + ")"}`);
      clearOverlays();
      return;
    }
    waveBoxes = r.boxes;
    log(`wave → ${r.boxes.length} box(es); click one to re-seed`);
    renderWaveBoxes();
  } finally {
    button.disabled = false;
  }
};

$("btn-record").onclick = async () => {
  const r = await post(recording ? "/api/record/stop" : "/api/record/start");
  log(`record ${recording ? "stop" : "start"} → ${r.message || (r.ok ? "ok" : "failed")}`);
};

/* Follow Start/Stop: each posts the selected group to the fixed-allowlist
   ProcessManager (audio + BT for vision+audio, +follow_server for with-nav).
   The follow-state panel then updates off the ws "follow" push. */
$("follow-start").addEventListener("click", async () => {
  const mode = selectedFollowMode();
  setFollowControlsRunning(true);
  const r = await post(`/api/proc/group/${mode}/start`, undefined);
  log(`follow start (${mode}) → ${r.error || "started"}`);
});
$("follow-stop").addEventListener("click", async () => {
  const mode = selectedFollowMode();
  const r = await post(`/api/proc/group/${mode}/stop`, undefined);
  log(`follow stop (${mode}) → ${r.error || "stopped"}`);
  setFollowControlsRunning(false);
});

/* Follow cruise settings: load current B/A/min-gap on startup, save on click.
   Persisted server-side to ~/.tk25/follow_cruise.yaml and injected as -p
   overrides on the next follow_server launch (no live update). */
async function loadCruise() {
  try {
    const c = await (await fetch("/api/follow/cruise")).json();
    $("cruise-goalgate").checked = !!c.enable_cruise_goalgate;
    $("cruise-carrot").checked = !!c.enable_cruise_carrot;
    $("cruise-min-gap").value = Number(c.cruise_min_gap).toFixed(2);
    $("cruise-status").textContent =
      `B=${c.enable_cruise_goalgate} A=${c.enable_cruise_carrot} gap=${c.cruise_min_gap} m`;
  } catch (e) { $("cruise-status").textContent = `load failed: ${e}`; }
}
$("cruise-save").addEventListener("click", async () => {
  const gap = parseFloat($("cruise-min-gap").value);
  if (isNaN(gap) || gap < 0.2 || gap > 1.5) {
    $("cruise-status").textContent = "Invalid min-gap (must be 0.2-1.5 m)";
    return;
  }
  const body = {
    enable_cruise_goalgate: $("cruise-goalgate").checked,
    enable_cruise_carrot: $("cruise-carrot").checked,
    cruise_min_gap: gap,
  };
  const r = await post("/api/follow/cruise", body);
  $("cruise-status").textContent = r.cruise_min_gap !== undefined
    ? `saved — B=${r.enable_cruise_goalgate} A=${r.enable_cruise_carrot} gap=${r.cruise_min_gap} m (restart follow to apply)`
    : `save failed: ${r.message || JSON.stringify(r)}`;
  log(`cruise save → ${JSON.stringify(r)}`);
});
loadCruise();

/* Stale banner + observer/bench mode chip + record state. */
setInterval(async () => {
  $("stale-banner").classList.toggle("hidden", Date.now() - lastStateAt < 1000);
  try {
    const st = await (await fetch("/api/status")).json();
    const m = st.goal.held ? ["bench", "on"] : st.goal.observer ? ["observer", "on"] : ["idle", "off"];
    lastMode = m[0];
    $("mode").textContent = m[0];
    $("mode").className = "badge " + m[1];
    recording = !!(st.recording && st.recording.active);
    const rb = $("btn-record");
    if (rb) {
      rb.textContent = recording ? "⏹ Stop recording" : "⏺ Record bag";
      rb.classList.toggle("rec-on", recording);
    }
  } catch (e) { /* status poll is best-effort */ }
}, 1000);

window.addEventListener("resize", renderWaveBoxes);
connectWS();
