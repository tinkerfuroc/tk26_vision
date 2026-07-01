// calibrate_web frontend. Vanilla JS, no build step.

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

// Canonical "level" park tilt (firmware deg), per-robot via calibration.yaml
// (tinker1=45, tinker2=30). Updated from /api/state; drives the Level jog
// button + the "physical tilt from level" grid display.
let LEVEL_TILT = 30;
function syncLevelButton() {
  const b = document.getElementById('btn-level');
  if (b) {
    b.dataset.tilt = String(LEVEL_TILT);
    b.textContent = `Level (0, +${LEVEL_TILT})`;
    b.title = `physical level (horizontal) — firmware tilt +${LEVEL_TILT}`;
  }
}

// FastAPI returns plain-text "Internal Server Error" on unhandled 500s, which
// crashes raw `r.json()` with "Unexpected token 'I'... is not valid JSON".
// Read once as text, JSON-parse if possible, otherwise wrap the text under
// `.detail` so callers' `body.detail` rendering still works.
async function readBody(r) {
  const text = await r.text();
  if (!text) return {};
  try { return JSON.parse(text); }
  catch { return { detail: text }; }
}
// Alias for older call sites that referenced a `readJsonResponse` helper
// that was never defined in this file. Same shape, same fallback.
const readJsonResponse = readBody;

// ---- side-panel tabs --------------------------------------------------------

function activateSideTab(name) {
  $$('.side-tab').forEach(b => b.classList.toggle('active', b.dataset.sideTab === name));
  $$('.side-panel-content').forEach(p => p.classList.toggle('active', p.id === 'side-' + name));
}
$$('.side-tab').forEach(b => b.addEventListener('click', () => activateSideTab(b.dataset.sideTab)));

// ---- resizable camera thumbnail --------------------------------------------
// Camera floats upper-left; controls wrap around it. A small handle in the
// camera's bottom-right corner lets the user drag to resize. Width persisted.
const CAM_HANDLE = $('#cam-resize');
const CAM_KEY = 'pt-calib-cam-width';
const MIN_CAM = 240, MAX_CAM = 720;

function clampCamWidth(px) {
  const maxAllowed = Math.min(MAX_CAM, window.innerWidth * 0.4);
  return Math.max(MIN_CAM, Math.min(maxAllowed, px));
}
function setCamWidth(px) {
  const w = clampCamWidth(px);
  document.documentElement.style.setProperty('--cam-w', w + 'px');
  return w;
}
(function initCamWidth() {
  const stored = parseInt(localStorage.getItem(CAM_KEY) || '', 10);
  setCamWidth(Number.isFinite(stored) ? stored : Math.min(480, window.innerWidth * 0.25));
})();
if (CAM_HANDLE) {
  let dragging = false, startX = 0, startW = 0;
  CAM_HANDLE.addEventListener('pointerdown', (e) => {
    dragging = true;
    startX = e.clientX;
    startW = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--cam-w'), 10) || 480;
    CAM_HANDLE.classList.add('dragging');
    try { CAM_HANDLE.setPointerCapture(e.pointerId); } catch (_) {}
    e.preventDefault();
  });
  CAM_HANDLE.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    setCamWidth(startW + (e.clientX - startX));
  });
  const endDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    CAM_HANDLE.classList.remove('dragging');
    try { CAM_HANDLE.releasePointerCapture(e.pointerId); } catch (_) {}
    const cur = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--cam-w'), 10);
    if (Number.isFinite(cur)) localStorage.setItem(CAM_KEY, String(cur));
  };
  CAM_HANDLE.addEventListener('pointerup', endDrag);
  CAM_HANDLE.addEventListener('pointercancel', endDrag);
}
window.addEventListener('resize', () => {
  const cur = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--cam-w'), 10);
  if (Number.isFinite(cur)) setCamWidth(cur);
});

// ---- camera refresh (MJPEG-ish polling) -------------------------------------
// Re-fetch /api/frame.jpg with cache-busting at ~3 Hz.
// `view-mode` radios toggle between the annotated (detection overlay) stream
// and the raw camera feed — useful when overlay encoding fails (e.g. missing
// camera_info) or when debugging why frames are black.
const IMG_MAIN = $('#camera-img');
const LIVE_FRAME = document.querySelector('.live-frame');

let rawMode = false;
let frameErrored = false;

function currentFrameUrl() {
  const base = '/api/frame.jpg';
  const params = new URLSearchParams({ t: String(Date.now()) });
  if (rawMode) params.set('raw', '1');
  return `${base}?${params.toString()}`;
}

function refreshFrame() {
  if (IMG_MAIN) IMG_MAIN.src = currentFrameUrl();
}
setInterval(refreshFrame, 330);

// Track whether the image actually loaded; toggle the placeholder overlay.
function markFrameErrored(val) {
  frameErrored = val;
  if (LIVE_FRAME) LIVE_FRAME.classList.toggle('no-frame', val);
}
if (IMG_MAIN) {
  IMG_MAIN.addEventListener('load', () => markFrameErrored(false));
  IMG_MAIN.addEventListener('error', () => markFrameErrored(true));
}

document.querySelectorAll('input[name="view-mode"]').forEach(r => {
  r.addEventListener('change', () => {
    rawMode = (document.querySelector('input[name="view-mode"]:checked').value === 'raw');
    refreshFrame();  // pick up the new URL without waiting for the next tick
  });
});

// ---- camera retarget --------------------------------------------------------

const BTN_RESUB = $('#btn-resubscribe');
if (BTN_RESUB) {
  BTN_RESUB.addEventListener('click', async () => {
    const topic = $('#topic-select').value;
    const status = $('#topic-status');
    if (!topic) {
      status.textContent = 'pick a topic first';
      status.className = 'status-line warn';
      return;
    }
    status.textContent = `subscribing to ${topic}…`;
    status.className = 'status-line warn';
    try {
      const r = await fetch('/api/camera/resubscribe', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ image_topic: topic }),
      });
      const body = await readBody(r);
      if (r.ok && body.ok) {
        status.textContent = `subscribed → image: ${body.image_topic}  info: ${body.camera_info_topic}`;
        status.className = 'status-line ok';
      } else {
        status.textContent = 'FAIL: ' + (body.detail || JSON.stringify(body));
        status.className = 'status-line err';
      }
    } catch (e) {
      status.textContent = 'ERROR: ' + e;
      status.className = 'status-line err';
    }
  });
}

// ---- WebSocket state --------------------------------------------------------

const INDICATOR = $('#conn-indicator');
let lastState = null;

// Auto-reconnecting WebSocket helper. Both the state-push (/ws) and the
// calibrate log fanout (/ws/calib-log) use this; only the message handler and
// optional indicator update differ.
function connectWS(path, {onMessage, onOpen, onClose, retryMs = 1500}) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(proto + '//' + location.host + path);
  ws.onopen = () => onOpen && onOpen(ws);
  ws.onclose = () => {
    onClose && onClose();
    setTimeout(() => connectWS(path, {onMessage, onOpen, onClose, retryMs}), retryMs);
  };
  ws.onerror = () => {/* onclose fires after */};
  ws.onmessage = (ev) => onMessage(ev, ws);
  return ws;
}

connectWS('/ws', {
  onOpen: () => {
    INDICATOR.textContent = 'WS: live';
    INDICATOR.classList.remove('dropped');
    INDICATOR.classList.add('connected');
  },
  onClose: () => {
    INDICATOR.textContent = 'WS: dropped — retrying';
    INDICATOR.classList.remove('connected');
    INDICATOR.classList.add('dropped');
  },
  onMessage: (ev) => {
    try { lastState = JSON.parse(ev.data); } catch (_) { return; }
    renderState(lastState);
  },
});

// ---- state rendering --------------------------------------------------------

function fmt(v, n = 4) {
  return (v === undefined || v === null) ? '–' : Number(v).toFixed(n);
}

function renderState(s) {
  if (typeof s.level_tilt_deg === 'number' && s.level_tilt_deg !== LEVEL_TILT) {
    LEVEL_TILT = s.level_tilt_deg;
    syncLevelButton();
  }
  $('#s-camera').textContent = s.have_camera ? 'streaming' : '—';
  $('#s-image-topic').textContent = s.image_topic || '—';
  $('#s-domain').textContent = s.ros_domain_id || '—';
  $('#s-frame-count').textContent = s.frame_count ?? 0;
  if (s.frame_age_sec === null || s.frame_age_sec === undefined) {
    $('#s-frame-age').textContent = '—';
  } else {
    const age = Number(s.frame_age_sec);
    $('#s-frame-age').textContent = age < 10 ? age.toFixed(2) + ' s' : '>10 s (stale)';
  }
  $('#s-frame-hz').textContent = s.frame_hz ? s.frame_hz.toFixed(1) + ' Hz' : '—';
  const phTopic = $('#placeholder-topic');
  if (phTopic) phTopic.textContent = s.image_topic || '/camera/color/image_raw';

  // Populate the topic dropdown — preserve current selection when possible.
  const sel = $('#topic-select');
  if (sel && Array.isArray(s.available_image_topics)) {
    const prev = sel.value;
    const topics = s.available_image_topics;
    const currentSub = s.image_topic || '';
    const preferred = prev || currentSub;
    const newOptions = topics.length === 0
      ? ['<option value="">(no Image topics on this ROS_DOMAIN_ID)</option>']
      : topics.map(t => `<option value="${t}"${t === preferred ? ' selected' : ''}>${t}${t === currentSub ? '  ← subscribed' : ''}</option>`);
    // Avoid rewriting on every tick if nothing changed (keeps focus/caret).
    const hash = newOptions.join('|');
    if (sel.dataset.hash !== hash) {
      sel.innerHTML = newOptions.join('');
      sel.dataset.hash = hash;
    }
  }
  $('#s-pt').textContent = s.have_pt_state ? (s.pt_connected ? 'connected' : 'disconnected') : '—';
  $('#s-tf').textContent = s.have_tf ? 'ok' : '—';
  $('#s-joints').textContent = s.have_xarm_joints
    ? `${s.xarm_joint_positions.length} joints`
    : '—';

  $('#s-pan').textContent  = fmt(s.pan_rad, 4) + ' rad (' + fmt(s.pan_rad * 180/Math.PI, 2) + '°)';
  $('#s-tilt').textContent = fmt(s.tilt_rad, 4) + ' rad (' + fmt(s.tilt_rad * 180/Math.PI, 2) + '°)';
  $('#s-ptok').textContent = s.pt_feedback_ok ? 'true' : 'false';

  const badge = $('#detection-badge');
  if (s.last_detection_ok) {
    badge.textContent = `corners=${s.last_detection_n_corners}  rms=${fmt(s.last_detection_rms, 2)}px  OK`;
    badge.style.color = '#5fd37f';
  } else {
    badge.textContent = `corners=${s.last_detection_n_corners}  NO DETECTION`;
    badge.style.color = '#ff7a7a';
  }

  if (s.t_base_ee) {
    $('#s-tmat').textContent = s.t_base_ee
      .map(row => row.map(v => (v >= 0 ? ' ' : '') + Number(v).toFixed(4)).join('  '))
      .join('\n');
  } else {
    $('#s-tmat').textContent = '—';
  }

  $('#s-safety').textContent = JSON.stringify(s.safety, null, 2);

  // --- xArm tab safety preview on current TF --------------------------------
  const ss = $('#xarm-safety-status');
  if (s.t_base_ee) {
    evaluateSafetyEnvelope(s.t_base_ee, s.safety, ss);
  } else {
    ss.textContent = 'waiting for TF…';
    ss.className = 'status-line warn';
  }

  // --- Pan-Tilt tab grid cheat-sheet + corner presets ----------------------
  renderGridFromState(s.grid);
}

let _gridSignature = null;
function renderGridFromState(grid) {
  if (!grid) return;
  const pan = grid.pan_deg || [];
  const tilt = grid.tilt_deg || [];
  const sig = pan.join(',') + '|' + tilt.join(',');
  if (sig === _gridSignature) return;        // only redraw when config changes
  _gridSignature = sig;
  const fmt = arr => arr.map(x => Number(x).toFixed(0)).join(', ');
  const panKv = $('#s-grid-pan');
  const tiltKv = $('#s-grid-tilt');
  const tiltPhysKv = $('#s-grid-tilt-phys');
  const nKv = $('#s-grid-n');
  if (panKv) panKv.textContent = pan.length ? `[${fmt(pan)}]°  (span ±${Math.max(...pan.map(Math.abs)).toFixed(0)}°)` : '—';
  if (tiltKv) tiltKv.textContent = tilt.length ? `[${fmt(tilt)}]°` : '—';
  if (tiltPhysKv) tiltPhysKv.textContent = tilt.length
    ? `[${fmt(tilt.map(t => t - LEVEL_TILT))}]°  (±${(Math.max(...tilt) - Math.min(...tilt)) / 2}° around level, + = up)`
    : '—';
  if (nKv) nKv.textContent = (pan.length * tilt.length) + ` (${pan.length} pan × ${tilt.length} tilt)`;
  renderGridCornerButtons(pan, tilt);
}

function renderGridCornerButtons(pan, tilt) {
  const root = $('#grid-corner-buttons');
  if (!root) return;
  root.innerHTML = '';
  if (!pan.length || !tilt.length) return;
  const pMin = Math.min(...pan), pMax = Math.max(...pan);
  const tMin = Math.min(...tilt), tMax = Math.max(...tilt);
  const tMid = tilt[Math.floor(tilt.length / 2)];
  const pMid = pan[Math.floor(pan.length / 2)];
  const corners = [
    [pMin, tMin, '⌜'], [pMax, tMin, '⌝'],
    [pMin, tMax, '⌞'], [pMax, tMax, '⌟'],
    [pMid, tMid, '⊙'],
  ];
  for (const [p, t, gly] of corners) {
    const b = document.createElement('button');
    b.textContent = `${gly} (${p}, ${t})`;
    b.title = `Physical tilt ${t - LEVEL_TILT}° from level (+ = up)`;
    b.dataset.pan = String(p);
    b.dataset.tilt = String(t);
    b.addEventListener('click', () => {
      $('#jog-pan').value = p;
      $('#jog-tilt').value = t;
      ptMove(p, t);
    });
    root.appendChild(b);
  }
}

function evaluateSafetyEnvelope(T, env, line) {
  const z = T[2][3];
  const dx = T[0][3] - env.mast_xy_center[0];
  const dy = T[1][3] - env.mast_xy_center[1];
  const r = Math.hypot(dx, dy);
  const parts = [];
  if (z < env.z_floor_m) parts.push(`z=${z.toFixed(3)} < floor ${env.z_floor_m}`);
  if (r < env.mast_radius_m && z < env.mast_z_max)
    parts.push(`xy inside mast (r=${r.toFixed(3)} < ${env.mast_radius_m})`);
  if (parts.length) {
    line.textContent = 'VIOLATION: ' + parts.join('; ');
    line.className = 'status-line err';
  } else {
    line.textContent = `safe (z=${z.toFixed(3)}, r_mast=${r.toFixed(3)})`;
    line.className = 'status-line ok';
  }
}

// ---- joint editor -----------------------------------------------------------

const N_JOINTS = 7;
const JOINT_INPUTS_EL = $('#joint-inputs');
const jointInputs = [];
for (let i = 0; i < N_JOINTS; i++) {
  const row = document.createElement('div');
  row.className = 'joint-row';
  row.innerHTML = `<label>J${i}</label><input type="number" step="0.01" value="0">`;
  JOINT_INPUTS_EL.appendChild(row);
  jointInputs.push(row.querySelector('input'));
}

function getUnit() {
  return document.querySelector('input[name="unit"]:checked').value;
}

function readJointsRad() {
  const unit = getUnit();
  return jointInputs.map(inp => {
    const v = parseFloat(inp.value) || 0;
    return unit === 'deg' ? v * Math.PI / 180 : v;
  });
}

function writeJoints(valuesRad) {
  const unit = getUnit();
  jointInputs.forEach((inp, i) => {
    const v = valuesRad[i] !== undefined ? valuesRad[i] : 0;
    inp.value = (unit === 'deg' ? (v * 180 / Math.PI) : v).toFixed(4);
  });
}

$('#btn-zero').addEventListener('click', () => writeJoints(Array(N_JOINTS).fill(0)));
$('#btn-load-current').addEventListener('click', () => {
  if (!lastState || !lastState.have_xarm_joints) {
    alert('xArm joint_states not yet received');
    return;
  }
  const p = lastState.xarm_joint_positions.slice(0, N_JOINTS);
  writeJoints(p);
});
document.querySelectorAll('input[name="unit"]').forEach(r => {
  r.addEventListener('change', () => {
    // Re-format current values in the newly-selected unit.
    const rad = readJointsRad();  // reads in OLD unit — no, reads in currently-selected unit, but we just changed it.
    // Hack: revert to rad via the non-selected unit mapping.
    // Simpler: track previous unit in a data-attribute.
  });
});
// Simplified unit-switch: just re-read current input values as the new unit.
// To keep values consistent across toggles, track prev unit and convert.
let prevUnit = 'rad';
document.querySelectorAll('input[name="unit"]').forEach(r => {
  r.addEventListener('change', () => {
    const nu = getUnit();
    if (nu === prevUnit) return;
    jointInputs.forEach(inp => {
      const v = parseFloat(inp.value) || 0;
      const asRad = prevUnit === 'deg' ? v * Math.PI / 180 : v;
      inp.value = (nu === 'deg' ? (asRad * 180 / Math.PI) : asRad).toFixed(4);
    });
    prevUnit = nu;
  });
});

// ---- xArm move --------------------------------------------------------------

$('#btn-xarm-move').addEventListener('click', async () => {
  const status = $('#xarm-move-status');
  const angles = readJointsRad();
  if (!confirm('Send xArm to these joints now?\n' + angles.map(a => a.toFixed(4)).join(', '))) {
    return;
  }
  status.textContent = 'moving…';
  status.className = 'status-line warn';
  try {
    const r = await fetch('/api/xarm/move', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ angles_rad: angles }),
    });
    const body = await readBody(r);
    if (r.ok && body.ok) {
      status.textContent = 'move complete: ' + body.message;
      status.className = 'status-line ok';
    } else {
      status.textContent = 'FAIL: ' + (body.message || body.detail || ('HTTP ' + r.status));
      status.className = 'status-line err';
    }
  } catch (e) {
    status.textContent = 'ERROR: ' + e;
    status.className = 'status-line err';
  }
});

// ---- Cartesian move (tinker_arm_msgs CartesianMove) -------------------------

function readCartPose() {
  return {
    translation: ['cart-x', 'cart-y', 'cart-z'].map(id => parseFloat($('#' + id).value) || 0),
    rotation:    ['cart-qx', 'cart-qy', 'cart-qz', 'cart-qw'].map(id => parseFloat($('#' + id).value) || 0),
  };
}

function writeCartPose(pose) {
  const ids = ['cart-x','cart-y','cart-z','cart-qx','cart-qy','cart-qz','cart-qw'];
  const vals = pose.translation.concat(pose.rotation);
  ids.forEach((id, i) => $('#' + id).value = (vals[i] || 0).toFixed(5));
}

$('#btn-cart-fill').addEventListener('click', () => {
  if (!lastState || !lastState.t_base_ee) {
    alert('T_base_ee not yet received');
    return;
  }
  const T = lastState.t_base_ee;
  // Extract translation directly; convert rotation matrix to quaternion.
  const tx = T[0][3], ty = T[1][3], tz = T[2][3];
  // Basic matrix->quaternion (standard algorithm). Works for any valid rotation.
  const m00 = T[0][0], m01 = T[0][1], m02 = T[0][2];
  const m10 = T[1][0], m11 = T[1][1], m12 = T[1][2];
  const m20 = T[2][0], m21 = T[2][1], m22 = T[2][2];
  const tr = m00 + m11 + m22;
  let qx, qy, qz, qw;
  if (tr > 0) {
    const s = 2 * Math.sqrt(tr + 1);
    qw = 0.25 * s;
    qx = (m21 - m12) / s;
    qy = (m02 - m20) / s;
    qz = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    qw = (m21 - m12) / s;
    qx = 0.25 * s;
    qy = (m01 + m10) / s;
    qz = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    qw = (m02 - m20) / s;
    qx = (m01 + m10) / s;
    qy = 0.25 * s;
    qz = (m12 + m21) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    qw = (m10 - m01) / s;
    qx = (m02 + m20) / s;
    qy = (m12 + m21) / s;
    qz = 0.25 * s;
  }
  writeCartPose({ translation: [tx, ty, tz], rotation: [qx, qy, qz, qw] });
});

$('#btn-cart-move').addEventListener('click', async () => {
  const status = $('#cart-move-status');
  const pose = readCartPose();
  const qn = Math.hypot(...pose.rotation);
  if (qn < 1e-3) { status.textContent = 'rotation quaternion is zero'; status.className = 'status-line err'; return; }
  // Normalise the quaternion before sending.
  pose.rotation = pose.rotation.map(v => v / qn);
  if (!confirm(`Cartesian move to t=(${pose.translation.map(v => v.toFixed(3)).join(',')}) `
             + `q=(${pose.rotation.map(v => v.toFixed(3)).join(',')})?`)) return;
  status.textContent = 'moving…';
  status.className = 'status-line warn';
  try {
    const r = await fetch('/api/xarm/move_cartesian', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ target_pose: pose }),
    });
    const body = await readBody(r);
    if (r.ok && body.ok) {
      status.textContent = 'move complete: ' + body.message;
      status.className = 'status-line ok';
    } else {
      status.textContent = 'FAIL: ' + (body.message || body.detail || ('HTTP ' + r.status));
      status.className = 'status-line err';
    }
  } catch (e) {
    status.textContent = 'ERROR: ' + e;
    status.className = 'status-line err';
  }
});

// ---- waypoint lists ---------------------------------------------------------

const PHASES = [
  {
    key: 'phase1_waypoints',
    label: 'Phase 1 — hand-eye (LEVEL, head horizontal)',
    hint: 'Head parks at (pan=0, tilt=+30 firmware) = physically level. Record 12–15 distinct xArm EE poses; aim for ≥60° orientation change between consecutive poses. Compact configs (wrist near base) minimise gravity sag. Gate: trans RMSE < 3 mm, rot RMSE < 0.5°.',
  },
  {
    key: 'phase1_waypoints_custom',
    label: 'Phase 1 — hand-eye (CUSTOM datasets)',
    customDatasets: true,
    hint: 'OPTIONAL extra hand-eye datasets. Add as many as you like — each has its own park (pan, tilt) and its own xArm poses that keep the marker in the camera FoV at that head pose. Pan range ±30°, tilt range 0..+30° firmware (where +30 = level, 0 = 30° down). Solving hand-eye on a custom set independently and comparing T_ee_marker vs the LEVEL solve is a powerful cross-check — they must agree to ~mm/°.',
  },
  {
    key: 'phase2_waypoints',
    label: 'Phase 2 — pan-tilt sweep anchors',
    hint: 'xArm frozen at each anchor, pan-tilt sweeps all 25 grid cells. Need 2–3 anchors with the board visible across every cell. Vary xArm Z by ≥10 cm between anchors so T_B_trans and T_ee_marker are disambiguated. Gate: held-out trans RMSE < 3 mm, rot RMSE < 0.4°.',
  },
  {
    key: 'sanity_xarm_angles_rad',
    label: 'Sanity pose (single)',
    hint: 'Recorded at session start and session end. Any repeatable pose works. If the start→end bracket disagrees by > 2 mm / 0.2°, something drifted (thermal, servo re-home, mount loosened) → flag the run.',
  },
];

const WP_ROOT = $('#waypoint-lists');
let wpState = {
  phase1_waypoints: [],
  phase2_waypoints: [], sanity_xarm_angles_rad: [],
};
// Custom hand-eye datasets: [{name, park_pan_deg, park_tilt_deg, waypoints}].
// Waypoints for dataset <name> are mirrored into wpState['phase1_waypoints_custom:<name>']
// so the generic add/remove/pushPhase machinery works unchanged.
const CUSTOM_PHASE_PREFIX = 'phase1_waypoints_custom:';
let customDatasets = [];

function renderWaypoints() {
  WP_ROOT.innerHTML = '';
  PHASES.forEach(phase => {
    const group = document.createElement('div');
    group.className = 'wp-group';

    // Custom datasets get a bespoke container (one sub-group per dataset).
    if (phase.customDatasets) {
      renderCustomDatasets(group, phase);
      WP_ROOT.appendChild(group);
      return;
    }

    const header = document.createElement('div');
    header.className = 'wp-header';
    header.innerHTML = `<strong>${phase.label}</strong>`;
    const add = document.createElement('button');
    add.textContent = '+ add current joints';
    add.addEventListener('click', () => addWaypoint(phase.key));
    header.appendChild(add);
    group.appendChild(header);

    if (phase.hint) {
      const hint = document.createElement('p');
      hint.className = 'muted wp-hint';
      hint.textContent = phase.hint;
      group.appendChild(hint);
    }

    group.appendChild(buildWaypointList(phase.key));
    WP_ROOT.appendChild(group);
  });
}

// Build the joint-list editor (rows + load/remove) for one phase key. Shared
// by the static phases and each custom dataset's sub-group.
function buildWaypointList(phaseKey) {
  const list = document.createElement('div');
  list.className = 'wp-list';
  const wps = wpState[phaseKey] || [];
  const items = (phaseKey === 'sanity_xarm_angles_rad')
    ? (wps.length > 0 && Array.isArray(wps[0]) ? wps : (wps.length ? [wps] : []))
    : wps;
  if (items.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'wp-item empty';
    empty.textContent = '(empty)';
    list.appendChild(empty);
  }
  items.forEach((wp, i) => {
    const row = document.createElement('div');
    row.className = 'wp-item';
    const txt = document.createElement('span');
    txt.textContent = `[${i}] ${wp.map(v => Number(v).toFixed(4)).join(', ')}`;
    row.appendChild(txt);
    const controls = document.createElement('span');
    const load = document.createElement('button');
    load.textContent = 'load';
    load.addEventListener('click', () => writeJoints(wp));
    const del = document.createElement('button');
    del.textContent = 'remove';
    del.addEventListener('click', () => removeWaypoint(phaseKey, i));
    controls.appendChild(load);
    controls.appendChild(del);
    row.appendChild(controls);
    list.appendChild(row);
  });
  return list;
}

async function fetchWaypoints() {
  try {
    const r = await fetch('/api/waypoints');
    if (r.ok) {
      // Merge static phases only — never clobber the custom dataset keys that
      // fetchCustomDatasets() seeds.
      Object.assign(wpState, await readBody(r));
      renderWaypoints();
    }
  } catch (e) {
    console.warn('fetchWaypoints failed', e);
  }
}
fetchWaypoints();

async function pushPhase(phase) {
  const body = { waypoints: wpState[phase] };
  // For sanity: server accepts either flat list or list-of-lists.
  if (phase === 'sanity_xarm_angles_rad'
      && wpState[phase].length
      && !Array.isArray(wpState[phase][0])) {
    body.waypoints = [wpState[phase]];
  }
  await fetch(`/api/waypoints/${phase}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
}

function addWaypoint(phase) {
  const cur = readJointsRad();
  if (phase === 'sanity_xarm_angles_rad') {
    wpState[phase] = cur;  // single pose replaces previous
  } else {
    (wpState[phase] = wpState[phase] || []).push(cur);
  }
  pushPhase(phase);
  renderWaypoints();
}

function removeWaypoint(phase, idx) {
  if (phase === 'sanity_xarm_angles_rad') {
    wpState[phase] = [];
  } else {
    wpState[phase].splice(idx, 1);
  }
  pushPhase(phase);
  renderWaypoints();
}

// ---- save -------------------------------------------------------------------

$('#btn-save').addEventListener('click', async () => {
  const status = $('#save-status');
  status.textContent = 'saving…';
  status.className = 'status-line warn';
  try {
    const r = await fetch('/api/waypoints/save', { method: 'POST' });
    const body = await readBody(r);
    if (r.ok && body.ok) {
      status.textContent = 'wrote ' + body.path;
      status.className = 'status-line ok';
    } else {
      status.textContent = 'FAIL: ' + (body.detail || JSON.stringify(body));
      status.className = 'status-line err';
    }
  } catch (e) {
    status.textContent = 'ERROR: ' + e;
    status.className = 'status-line err';
  }
});

async function loadBoardSpec() {
  try {
    const r = await fetch('/api/board');
    if (!r.ok) return;
    const b = await readBody(r);
    const set = (id, v) => { const el = $('#' + id); if (el) el.textContent = v; };
    set('s-board-grid',    `${b.squares_x} × ${b.squares_y} squares`);
    set('s-board-square',  `${(b.square_len_m * 1000).toFixed(1)} mm`);
    set('s-board-marker',  `${(b.marker_len_m * 1000).toFixed(1)} mm`);
    set('s-board-dict',    b.dict);
    set('s-board-corners', String(b.inner_corners));
    set('s-board-size',    `${(b.board_size_m[0] * 1000).toFixed(0)} × ${(b.board_size_m[1] * 1000).toFixed(0)} mm`);
    // Compact one-line summary for the Calibrate tab.
    const calibLine = $('#calib-board-spec');
    if (calibLine) {
      calibLine.textContent =
        `${b.squares_x}×${b.squares_y} ${b.dict}, ` +
        `square=${(b.square_len_m * 1000).toFixed(1)}mm, ` +
        `marker=${(b.marker_len_m * 1000).toFixed(1)}mm, ` +
        `physical=${(b.board_size_m[0] * 1000).toFixed(0)}×${(b.board_size_m[1] * 1000).toFixed(0)}mm`;
    }
  } catch (e) { /* server not up yet */ }
}
loadBoardSpec();

// ---- Phase-1 custom hand-eye datasets ---------------------------------------

function _customKey(name) { return CUSTOM_PHASE_PREFIX + name; }

async function fetchCustomDatasets() {
  try {
    const r = await fetch('/api/calib/custom_datasets');
    if (!r.ok) return;
    const body = await readBody(r);
    customDatasets = body.datasets || [];
    // Mirror each dataset's waypoints into wpState so add/remove/pushPhase work.
    customDatasets.forEach(d => { wpState[_customKey(d.name)] = d.waypoints || []; });
    renderWaypoints();
    // The Calibrate-tab pickers (collect/handeye-custom + chain/polish) depend
    // on the dataset list too.
    populateCustomDatasetPicker();
    rebuildHandeyeSelectors();
  } catch (e) { console.warn('fetchCustomDatasets failed', e); }
}

async function addCustomDataset() {
  const raw = prompt('New custom dataset name (letters/digits/underscore, e.g. high_shelf):', '');
  if (raw === null) return;
  const name = raw.trim();
  if (!name) return;
  try {
    const r = await fetch('/api/calib/custom_datasets', {
      method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify({ name }),
    });
    const body = await readBody(r);
    if (!r.ok) { alert('Could not add dataset: ' + (body.detail || JSON.stringify(body))); return; }
    await fetchCustomDatasets();
  } catch (e) { alert('add dataset failed: ' + e); }
}

async function removeCustomDataset(name) {
  if (!confirm(`Remove custom dataset "${name}" and all its waypoints?`)) return;
  try {
    const r = await fetch('/api/calib/custom_datasets/' + encodeURIComponent(name), { method: 'DELETE' });
    if (!r.ok) { const b = await readBody(r); alert('remove failed: ' + (b.detail || JSON.stringify(b))); return; }
    delete wpState[_customKey(name)];
    await fetchCustomDatasets();
  } catch (e) { alert('remove dataset failed: ' + e); }
}

async function saveCustomPark(name, pan_deg, tilt_deg, statusEl) {
  try {
    const r = await fetch('/api/calib/custom_datasets/' + encodeURIComponent(name) + '/park', {
      method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify({ pan_deg, tilt_deg }),
    });
    const body = await readBody(r);
    if (statusEl) {
      statusEl.textContent = r.ok ? `saved: pan=${pan_deg}°, tilt=${tilt_deg}°`
                                  : ('save failed: ' + (body.detail || ''));
      statusEl.className = r.ok ? 'status-line ok' : 'status-line err';
    }
    if (r.ok) {
      const d = customDatasets.find(x => x.name === name);
      if (d) { d.park_pan_deg = pan_deg; d.park_tilt_deg = tilt_deg; }
      populateCustomDatasetPicker();
    }
  } catch (e) {
    if (statusEl) { statusEl.textContent = 'save failed: ' + e; statusEl.className = 'status-line err'; }
  }
}

// Render the whole "Phase 1 — hand-eye (CUSTOM datasets)" container: an add
// button, the hint, then one sub-group per dataset.
function renderCustomDatasets(group, phase) {
  const header = document.createElement('div');
  header.className = 'wp-header';
  header.innerHTML = `<strong>${phase.label}</strong>`;
  const add = document.createElement('button');
  add.textContent = '+ add custom dataset';
  add.addEventListener('click', addCustomDataset);
  header.appendChild(add);
  group.appendChild(header);

  if (phase.hint) {
    const hint = document.createElement('p');
    hint.className = 'muted wp-hint';
    hint.textContent = phase.hint;
    group.appendChild(hint);
  }

  if (!customDatasets.length) {
    const empty = document.createElement('p');
    empty.className = 'muted wp-hint';
    empty.textContent = '(no custom datasets — click "+ add custom dataset")';
    group.appendChild(empty);
    return;
  }

  customDatasets.forEach(d => {
    const sub = document.createElement('div');
    sub.className = 'wp-custom-dataset';

    const subHeader = document.createElement('div');
    subHeader.className = 'wp-header';
    subHeader.innerHTML = `<strong>↳ ${d.name}</strong>`;
    const addWp = document.createElement('button');
    addWp.textContent = '+ add current joints';
    addWp.addEventListener('click', () => addWaypoint(_customKey(d.name)));
    subHeader.appendChild(addWp);
    const rm = document.createElement('button');
    rm.textContent = 'remove dataset';
    rm.className = 'danger';
    rm.addEventListener('click', () => removeCustomDataset(d.name));
    subHeader.appendChild(rm);
    sub.appendChild(subHeader);

    // Per-dataset park controls.
    const park = document.createElement('div');
    park.className = 'wp-custom-park';
    const panInput = document.createElement('input');
    panInput.type = 'number'; panInput.step = '0.5'; panInput.min = '-30'; panInput.max = '30';
    panInput.value = d.park_pan_deg;
    const tiltInput = document.createElement('input');
    tiltInput.type = 'number'; tiltInput.step = '0.5'; tiltInput.min = '0'; tiltInput.max = '45';
    tiltInput.value = d.park_tilt_deg;
    const panLbl = document.createElement('label'); panLbl.append('Park pan: ', panInput, ' °');
    const tiltLbl = document.createElement('label'); tiltLbl.append('Park tilt: ', tiltInput, ' °');
    const saveBtn = document.createElement('button'); saveBtn.type = 'button'; saveBtn.textContent = 'Save park';
    const status = document.createElement('span'); status.className = 'status-line muted';
    saveBtn.addEventListener('click', () => {
      const pan = parseFloat(panInput.value), tilt = parseFloat(tiltInput.value);
      if (Number.isNaN(pan) || Number.isNaN(tilt)) {
        status.textContent = 'invalid number'; status.className = 'status-line err'; return;
      }
      saveCustomPark(d.name, pan, tilt, status);
    });
    park.append(panLbl, tiltLbl, saveBtn, status);
    sub.appendChild(park);

    sub.appendChild(buildWaypointList(_customKey(d.name)));
    group.appendChild(sub);
  });
}

fetchCustomDatasets();

async function loadWaypointPaths() {
  try {
    const r = await fetch('/api/waypoints/paths');
    if (!r.ok) return;
    const body = await readBody(r);
    const setCell = (id, v) => { const el = $('#' + id); if (el) el.textContent = v || '(unresolved)'; };
    setCell('wp-path-draft', body.draft);
    setCell('wp-path-promote', body.promote);
  } catch (e) { /* tab may not be in DOM yet */ }
}
loadWaypointPaths();

const BTN_PROMOTE = $('#btn-promote');
if (BTN_PROMOTE) BTN_PROMOTE.addEventListener('click', async () => {
  const status = $('#promote-status');
  const promoteTarget = $('#wp-path-promote')?.textContent || 'calibration.yaml';
  if (!confirm(`Overwrite ${promoteTarget} with the current waypoints?\n\nThe existing file will be renamed to <stem>.yaml.old-<timestamp> first.`)) {
    return;
  }
  status.textContent = 'promoting…';
  status.className = 'status-line warn';
  try {
    const r = await fetch('/api/waypoints/promote', { method: 'POST' });
    const body = await readBody(r);
    if (r.ok && body.ok) {
      status.textContent = 'wrote ' + body.path + (body.backup ? '  (backup: ' + body.backup + ')' : '  (no prior file existed)');
      status.className = 'status-line ok';
    } else {
      status.textContent = 'FAIL: ' + (body.detail || JSON.stringify(body));
      status.className = 'status-line err';
    }
  } catch (e) {
    status.textContent = 'ERROR: ' + e;
    status.className = 'status-line err';
  }
});

const BTN_DEDUPE = $('#btn-dedupe');
if (BTN_DEDUPE) BTN_DEDUPE.addEventListener('click', async () => {
  const status = $('#dedupe-status');
  if (!confirm('Drop near-duplicate waypoints from the in-memory lists?\n\nThe change is in-memory only -- click Save or Promote afterward to persist.')) {
    return;
  }
  status.textContent = 'deduping…';
  status.className = 'status-line warn';
  try {
    const r = await fetch('/api/waypoints/dedupe', { method: 'POST' });
    const body = await readBody(r);
    if (r.ok && body.ok) {
      const removed = body.removed || {};
      const entries = Object.entries(removed);
      if (entries.length === 0) {
        status.textContent = 'no duplicates found';
      } else {
        const summary = entries.map(([k, v]) => `${k}=${v}`).join(', ');
        status.textContent = `removed: ${summary}  (Save or Promote to persist)`;
      }
      status.className = 'status-line ok';
      await fetchWaypoints();
    } else {
      status.textContent = 'FAIL: ' + (body.detail || JSON.stringify(body));
      status.className = 'status-line err';
    }
  } catch (e) {
    status.textContent = 'ERROR: ' + e;
    status.className = 'status-line err';
  }
});

async function _reloadWaypointsFromSource(source, confirmMsg) {
  const status = $('#reload-status');
  if (confirmMsg && !confirm(confirmMsg)) return;
  status.textContent = `reloading from ${source}…`;
  status.className = 'status-line warn';
  try {
    const r = await fetch('/api/waypoints/reload', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ source }),
    });
    // FastAPI returns plain-text "Internal Server Error" on unhandled 500s.
    // Read once as text, then try JSON; otherwise wrap the text so the
    // real backend message lands in the status line.
    const text = await r.text();
    let body;
    try { body = text ? JSON.parse(text) : {}; }
    catch { body = { detail: text }; }
    if (r.ok && body.ok) {
      const counts = Object.entries(body.counts || {})
        .map(([k, v]) => `${k}=${v}`).join(', ') || '(no waypoint sections)';
      status.textContent = `loaded from ${body.path}  (${counts})`;
      status.className = 'status-line ok';
      await fetchWaypoints();
    } else {
      status.textContent = `FAIL [${r.status}]: ` + (body.detail || JSON.stringify(body));
      status.className = 'status-line err';
    }
  } catch (e) {
    status.textContent = 'ERROR: ' + e;
    status.className = 'status-line err';
  }
}

const BTN_RELOAD_DRAFT = $('#btn-reload-draft');
if (BTN_RELOAD_DRAFT) BTN_RELOAD_DRAFT.addEventListener('click', () =>
  _reloadWaypointsFromSource('draft',
    'Replace in-memory waypoints with the contents of the draft yaml? '
    + 'Any unsaved edits will be lost.'));

const BTN_RELOAD_CONFIG = $('#btn-reload-config');
if (BTN_RELOAD_CONFIG) BTN_RELOAD_CONFIG.addEventListener('click', () =>
  _reloadWaypointsFromSource('promote',
    'Replace in-memory waypoints with the contents of the source-tree '
    + 'calibration.yaml? Any unsaved edits will be lost.'));

// ---- pan-tilt jog -----------------------------------------------------------

async function ptMove(panDeg, tiltDeg) {
  const status = $('#pt-move-status');
  status.textContent = `moving pan=${panDeg} tilt=${tiltDeg}…`;
  status.className = 'status-line warn';
  try {
    const r = await fetch('/api/pantilt/move', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ pan_deg: panDeg, tilt_deg: tiltDeg }),
    });
    const body = await readBody(r);
    status.textContent = body.message || 'ok';
    status.className = 'status-line ok';
  } catch (e) {
    status.textContent = 'ERROR: ' + e;
    status.className = 'status-line err';
  }
}

$('#btn-pt-move').addEventListener('click', () => {
  const p = parseFloat($('#jog-pan').value) || 0;
  const t = parseFloat($('#jog-tilt').value) || 0;
  ptMove(p, t);
});
$$('.jog-buttons button').forEach(b => {
  b.addEventListener('click', () => {
    const p = parseFloat(b.dataset.pan);
    const t = parseFloat(b.dataset.tilt);
    $('#jog-pan').value = p;
    $('#jog-tilt').value = t;
    ptMove(p, t);
  });
});

// Zero-state wizard (firmware: T:501 raw=1 new=2, then T:502 id=1, T:502 id=2).
// Split across two server endpoints because the operator must physically
// disconnect and reconnect motor 2 between the two firmware writes. The state
// machine here just gates the buttons and posts on advance.
//
//   idle ── start ──► unplug ── /remap ──► reconnect ── /finalize ──► done
//                                ▲ (firmware IDs mutated past this point)
//                                │ aborting strands the chain half-configured
const ZW_STEPS = ['idle', 'unplug', 'reconnect', 'done'];

function zwShow(stepName) {
  for (const s of ZW_STEPS) {
    const el = document.getElementById(`zw-${s}`);
    if (el) el.hidden = (s !== stepName);
  }
}

function zwSetStatus(text, cls) {
  const status = $('#pt-set-zero-status');
  status.textContent = text;
  status.className = 'status-line ' + (cls || '');
}

async function zwPost(url, label) {
  zwSetStatus(`${label}…`, 'warn');
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: '{}',
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      zwSetStatus(`FAILED (HTTP ${r.status}): ${body.detail || body.message || r.statusText}`, 'err');
      return { ok: false };
    }
    if (!body.ok) {
      zwSetStatus(`FAILED: ${body.message || '(no message)'}`, 'err');
      return { ok: false };
    }
    zwSetStatus(`${label} OK: ${body.message || ''}`, 'ok');
    return { ok: true };
  } catch (e) {
    zwSetStatus(`ERROR: ${e}`, 'err');
    return { ok: false };
  }
}

$('#btn-zw-start').addEventListener('click', () => {
  if (!confirm(
    'Start the firmware zero-state wizard?\n\n'
    + 'The CURRENT physical pan-tilt pose will become the new firmware zero. '
    + 'Make sure the pan-tilt is already jogged to where you want (0, 0).\n\n'
    + 'Existing pan-tilt calibration will be invalidated.'
  )) return;
  zwSetStatus('wizard started — waiting for motor 2 to be unplugged.', 'warn');
  zwShow('unplug');
});

$('#btn-zw-cancel-1').addEventListener('click', () => {
  zwSetStatus('wizard cancelled (no firmware writes were made).', '');
  zwShow('idle');
});

$('#btn-zw-unplug-done').addEventListener('click', async () => {
  const btn = $('#btn-zw-unplug-done');
  btn.disabled = true;
  const res = await zwPost('/api/pantilt/zero_wizard/remap', 'sending T:501 raw=1 new=2');
  btn.disabled = false;
  if (res.ok) {
    zwShow('reconnect');
  }
  // On failure stay on the unplug step so the operator can retry.
});

$('#btn-zw-cancel-2').addEventListener('click', () => {
  if (!confirm(
    'Abort the wizard now?\n\n'
    + 'T:501 has already been sent, so the servo ID mapping is in a transitional '
    + 'state. To recover you must either re-run the wizard end-to-end, or power-cycle '
    + 'the servo bus and reset IDs manually.'
  )) return;
  zwSetStatus('wizard aborted AFTER T:501 — servo ID mapping is half-applied.', 'err');
  zwShow('idle');
});

$('#btn-zw-reconnect-done').addEventListener('click', async () => {
  const btn = $('#btn-zw-reconnect-done');
  btn.disabled = true;
  const res = await zwPost('/api/pantilt/zero_wizard/finalize', 'sending T:502 id=1 then id=2');
  btn.disabled = false;
  if (res.ok) {
    zwShow('done');
    zwSetStatus('zero state stored on both servos.', 'ok');
  }
});

$('#btn-zw-restart').addEventListener('click', () => {
  zwSetStatus('', '');
  zwShow('idle');
});

// ============================================================================
// Calibrate tab
// ============================================================================
//
// Drives the post-collection CLI stack (`run_calibration.py` +
// `apply_to_urdf.py`) through the server-side CalibrateRunner. Everything
// here is file-I/O; nothing in this block ever moves the robot.

const CALIB = {
  currentSession: null,
  sessionsDir: null,
  urdfTargets: [],
  activeRunId: null,
  residualSource: 'handeye',
  // filename -> parsed JSON, scoped to the currently selected session.
  // Cleared on session change so we never accumulate cross-session data.
  fileCache: {},
  prereqs: {},        // cmd -> required session-relative filenames
  collectEnabled: true,
  // Last session-detail files map; used by the dataset-selector change
  // handlers to re-evaluate run-button enablement without a server round-trip.
  lastFiles: {},
};

const CALIB_LOG = $('#calib-log');
const CALIB_SESSION_SELECT = $('#calib-session-select');
const CALIB_SESSION_STATUS = $('#calib-session-status');
const CALIB_CANCEL_BTN = $('#calib-cancel-btn');

// ---- session discovery -----------------------------------------------------

async function calibLoadSessions() {
  try {
    const r = await fetch('/api/calib/sessions');
    const body = await readJsonResponse(r);
    CALIB.sessionsDir = body.sessions_dir;
    const sdirEl = $('#calib-sessions-dir');
    if (sdirEl) sdirEl.textContent = body.sessions_dir || '—';
    const prev = CALIB_SESSION_SELECT.value;
    const opts = ['<option value="">(none)</option>']
      .concat((body.sessions || []).map(s => `<option value="${s.name}">${s.name}  (${s.files.length} files)</option>`));
    CALIB_SESSION_SELECT.innerHTML = opts.join('');
    const names = (body.sessions || []).map(s => s.name);
    if (prev && names.includes(prev)) {
      CALIB_SESSION_SELECT.value = prev;
    } else if (names.length > 0) {
      CALIB_SESSION_SELECT.value = names[names.length - 1];
    }
    if (CALIB_SESSION_SELECT.value) {
      CALIB.currentSession = CALIB_SESSION_SELECT.value;
      await calibLoadSessionDetail(CALIB.currentSession);
    }
  } catch (e) {
    CALIB_SESSION_STATUS.textContent = 'ERROR listing sessions: ' + e;
    CALIB_SESSION_STATUS.className = 'status-line err';
  }
}

CALIB_SESSION_SELECT.addEventListener('change', async () => {
  const next = CALIB_SESSION_SELECT.value || null;
  if (next === CALIB.currentSession) return;
  CALIB.currentSession = next;
  CALIB.fileCache = {};
  if (CALIB.currentSession) {
    await calibLoadSessionDetail(CALIB.currentSession);
  } else {
    calibRenderFiles(null); calibRenderGates([]); calibRenderParams(null); calibRenderResiduals();
  }
});

$('#calib-new-session-btn').addEventListener('click', async () => {
  const name = $('#calib-new-session-name').value.trim();
  if (!name) {
    CALIB_SESSION_STATUS.textContent = 'enter a name first';
    CALIB_SESSION_STATUS.className = 'status-line warn';
    return;
  }
  try {
    const r = await fetch('/api/calib/session', {
      method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify({name}),
    });
    const body = await readJsonResponse(r);
    if (!r.ok) throw new Error(body.detail || JSON.stringify(body));
    CALIB_SESSION_STATUS.textContent = 'created ' + body.path;
    CALIB_SESSION_STATUS.className = 'status-line ok';
    $('#calib-new-session-name').value = '';
    await calibLoadSessions();
    CALIB_SESSION_SELECT.value = name;
    CALIB_SESSION_SELECT.dispatchEvent(new Event('change'));
  } catch (e) {
    CALIB_SESSION_STATUS.textContent = 'FAIL: ' + e.message;
    CALIB_SESSION_STATUS.className = 'status-line err';
  }
});

// ---- session detail --------------------------------------------------------

async function calibLoadSessionDetail(name) {
  const r = await fetch('/api/calib/session/' + encodeURIComponent(name));
  const body = await readJsonResponse(r);
  if (!r.ok) throw new Error(body.detail || JSON.stringify(body));
  const files = body.files || {};
  CALIB.lastFiles = files;
  calibRenderFiles(files);
  calibRenderGates(body.gates || []);
  // Ensure the handeye/phase1 selectors list every custom dataset (covers the
  // case where a session is selected before fetchCustomDatasets resolved).
  rebuildHandeyeSelectors();
  calibRenderDatasetSelectors(files);
  calibApplyRunEnablement(files);
  // polish.json takes precedence over chain.json when both are present.
  const paramsSource = ['polish.json', 'chain.json'].find(f => files[f]?.exists) || null;
  // Run params fetch + residuals + coverage in parallel; they're independent.
  await Promise.all([
    paramsSource
      ? calibFetchFile(name, paramsSource).then(j => calibRenderParams({file: paramsSource, data: j}))
      : Promise.resolve(calibRenderParams(null)),
    calibRenderResiduals(),
    calibRenderCoverage(),
    files['validation.json']?.exists
      ? calibFetchFile(name, 'validation.json').then(calibRenderValidate)
      : Promise.resolve(calibRenderValidate(null)),
  ]);
}

function calibRenderValidate(j) {
  const panel = document.getElementById('calib-validate-panel');
  if (!panel) return;
  if (!j) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const verdict = j.verdict || '—';
  const pill = document.getElementById('calib-validate-verdict');
  if (pill) {
    pill.textContent = verdict;
    pill.className = 'calib-verdict-pill calib-verdict-' + verdict.toLowerCase();
  }
  const sc = j.self_consistency || {};
  const summary = document.getElementById('calib-validate-summary');
  if (summary) {
    const rmse_t = (sc.trans_rmse_m ?? 0) * 1000;
    const rmse_r = (sc.rot_rmse_rad ?? 0) * 180 / Math.PI;
    summary.textContent = ` n=${j.n_samples_used}/${j.n_samples_total}  ·  ` +
                          `trans_rmse ${fmt(rmse_t, 2)} mm  ·  ` +
                          `rot_rmse ${fmt(rmse_r, 3)}°`;
  }
  const tbl = document.getElementById('calib-validate-metrics');
  if (!tbl) return;
  const std = sc.trans_std_xyz_m || [0, 0, 0];
  const rows = [
    ['trans max',   fmt((sc.trans_max_m ?? 0) * 1000, 2) + ' mm'],
    ['rot max',     fmt((sc.rot_max_rad ?? 0) * 180 / Math.PI, 3) + '°'],
    ['trans std X', fmt(std[0] * 1000, 2) + ' mm'],
    ['trans std Y', fmt(std[1] * 1000, 2) + ' mm'],
    ['trans std Z', fmt(std[2] * 1000, 2) + ' mm'],
  ];
  const th = j.thresholds || {};
  rows.push(['thresholds (PASS / WARN)',
             `${fmt(th.trans_pass_mm ?? 5, 1)} mm / ${fmt(th.rot_pass_deg ?? 0.5, 2)}°  ·  ` +
             `${fmt(th.trans_warn_mm ?? 10, 1)} mm / ${fmt(th.rot_warn_deg ?? 1, 2)}°`]);
  tbl.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
}

async function calibFetchFile(session, filename) {
  if (CALIB.fileCache[filename]) return CALIB.fileCache[filename];
  const r = await fetch(`/api/calib/session/${encodeURIComponent(session)}/file/${filename}`);
  if (!r.ok) return null;
  const body = await readJsonResponse(r);
  CALIB.fileCache[filename] = body;
  return body;
}

// ---- files table -----------------------------------------------------------

const CALIB_TRACKED_FILES = [
  {name: 'phase1_handeye.json',        kind: 'collector'},
  {name: 'phase1_handeye_custom.json', kind: 'collector'},
  {name: 'phase2_chain.json',          kind: 'collector'},
  {name: 'sanity.json',                kind: 'collector'},
  {name: 'phase4_validation.json',     kind: 'collector'},
  {name: 'handeye.json',               kind: 'analyser'},
  {name: 'handeye_custom.json',        kind: 'analyser'},
  {name: 'chain.json',                 kind: 'analyser'},
  {name: 'polish.json',                kind: 'analyser'},
  {name: 'validation.json',            kind: 'analyser'},
];

// Backend session-detail returns mtimes as Unix seconds; format them for the
// dataset-selector labels so an operator who forgot to re-collect sees the
// staleness immediately.
function _calibRelTime(unixSec) {
  if (!unixSec) return '';
  const dt = (Date.now() / 1000) - unixSec;
  if (dt < 90) return 'just now';
  if (dt < 3600) return `${Math.round(dt/60)}m ago`;
  if (dt < 86400) return `${Math.round(dt/3600)}h ago`;
  return new Date(unixSec * 1000).toLocaleDateString();
}

function calibRenderFiles(files) {
  const root = $('#calib-files-table');
  if (!files) {
    root.innerHTML = '<tr><td colspan="2" class="muted">(pick a session above)</td></tr>';
    return;
  }
  const rows = CALIB_TRACKED_FILES.map(({name, kind}) => {
    const info = files[name] || {};
    let rhs;
    if (!info.exists) {
      rhs = '<span class="muted">not yet</span>';
    } else {
      const parts = [];
      if ('n_samples' in info) parts.push(info.n_samples + ' samples');
      if ('trans_rmse_m' in info) parts.push(fmt(info.trans_rmse_m * 1000, 2) + ' mm');
      if ('rot_rmse_rad' in info) parts.push(fmt(info.rot_rmse_rad * 180 / Math.PI, 3) + '°');
      if ('val_trans_rmse_m' in info) parts.push('val ' + fmt(info.val_trans_rmse_m * 1000, 2) + ' mm');
      if ('val_rot_rmse_rad' in info) parts.push('val ' + fmt(info.val_rot_rmse_rad * 180 / Math.PI, 3) + '°');
      if (info.mtime) parts.push(new Date(info.mtime * 1000).toLocaleTimeString());
      rhs = '<span style="color:var(--ok)">✓</span>  ' + parts.join('  ·  ');
    }
    return `<tr><td>${name}</td><td>${rhs}</td></tr>`;
  });
  root.innerHTML = rows.join('');
}

// ---- gates table -----------------------------------------------------------

// Format a numeric gate value/threshold by its declared unit. Server reports
// SI internally (m, rad) so we apply the *1000 / rad->deg conversion here.
function gateFmt(value, unit, decimals = 3) {
  if (unit === 'mm') return fmt(value * 1000, decimals) + ' mm';
  if (unit === 'deg') return fmt(value * 180 / Math.PI, decimals) + '°';
  return fmt(value, decimals) + ' ' + unit;
}

function calibRenderGates(gates) {
  const root = $('#calib-gates');
  if (!gates || gates.length === 0) {
    root.innerHTML = '<tr><td colspan="3" class="muted">(run handeye / chain to populate)</td></tr>';
    return;
  }
  const rows = gates.map(g => {
    const value = ('value' in g) ? gateFmt(g.value, g.unit, 3) : '—';
    const thresh = '< ' + gateFmt(g.threshold, g.unit, 2);
    const statusClass = 'gate-' + g.status;
    const statusText = g.status.toUpperCase();
    return `<tr><td>${g.label}</td><td>${value} (${thresh})</td><td class="${statusClass}">${statusText}</td></tr>`;
  });
  root.innerHTML = rows.join('');
}

// ---- params table ----------------------------------------------------------

function calibRenderParams(info) {
  const root = $('#calib-params');
  if (!info) {
    root.innerHTML = '<tr><td colspan="2" class="muted">(run chain / polish to populate)</td></tr>';
    return;
  }
  const p = info.data.params || {};
  const tra = v => (v === undefined ? '—' : `[${v.map(x => Number(x).toFixed(4)).join(', ')}]`);
  const rows = [
    ['source', info.file],
    ['T_A translation', tra(p.t_a)],
    ['T_B translation', tra(p.t_b_trans)],
    ['T_B rotvec (rad)', tra(p.t_b_rotvec)],
    ['T_ee_marker trans', tra(p.t_ee_marker_trans)],
    ['T_ee_marker rotvec (rad)', tra(p.t_ee_marker_rotvec)],
    ['θ_t_offset', p.theta_t_offset_deg !== undefined ? `${p.theta_t_offset_deg.toFixed(3)}°` : '—'],
    ['θ_p_offset', p.theta_p_offset_deg !== undefined ? `${p.theta_p_offset_deg.toFixed(3)}°` : '—'],
    ['L_pan', p.l_pan !== undefined ? `${p.l_pan.toFixed(4)} m` : '—'],
  ];
  root.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
}

// ---- run subprocess --------------------------------------------------------

const CALIB_LOG_MAX_LINES = 2000;
function calibLogAppend(text, className) {
  if (!CALIB_LOG) return;
  if (CALIB_LOG.textContent === '(no runs yet)') CALIB_LOG.textContent = '';
  const span = document.createElement('span');
  if (className) span.className = className;
  span.textContent = text + '\n';
  CALIB_LOG.appendChild(span);
  // Bound the DOM size so a 20-min Phase-2 sweep doesn't drown the browser.
  while (CALIB_LOG.childElementCount > CALIB_LOG_MAX_LINES) {
    CALIB_LOG.removeChild(CALIB_LOG.firstChild);
  }
  CALIB_LOG.scrollTop = CALIB_LOG.scrollHeight;
}

async function calibRun(cmd, flags) {
  if (!CALIB.currentSession) {
    CALIB_SESSION_STATUS.textContent = 'pick a session first';
    CALIB_SESSION_STATUS.className = 'status-line warn';
    return;
  }
  // collect_phase1_custom + handeye_custom act on the dataset chosen in the
  // Calibrate-tab picker. Resolve it up front so both the confirm text and the
  // request body see the same name.
  let customName = '';
  if (cmd === 'collect_phase1_custom' || cmd === 'handeye_custom') {
    const dsel = document.getElementById('calib-custom-dataset-select');
    customName = dsel ? dsel.value : '';
    if (!customName) {
      CALIB_SESSION_STATUS.textContent = 'no custom dataset selected — add one in the xArm Waypoints tab';
      CALIB_SESSION_STATUS.className = 'status-line warn';
      return;
    }
  }
  // Collection commands move the physical robot -- require an explicit confirm
  // so a stray click doesn't kick off a 20-minute sweep.
  if (cmd.startsWith('collect_')) {
    const cd = customName ? customDatasets.find(d => d.name === customName) : null;
    const human = {
      collect_phase1:        'Phase 1 LEVEL (pan=0, head horizontal, uses phase1_waypoints)',
      collect_phase1_custom: cd
        ? `Phase 1 CUSTOM "${customName}" (pan=${cd.park_pan_deg}°, tilt=${cd.park_tilt_deg}°)`
        : `Phase 1 CUSTOM "${customName}"`,
      collect_dry_run:       'Dry-run waypoint validation (NO image capture, NO pan-tilt motion). Sends each xArm waypoint via JointMove and reports which ones fail. Cheap; safe to run before a full collect.',
      collect_phase2:        'Phase 2 (pan-tilt grid sweep at each xArm anchor)',
      collect_sanity:        'sanity pose (single pose)',
      collect_both:          'the full run (Phase 1 + Phase 2 + sanity; ~20-30 min)',
    }[cmd] || cmd;
    const yes = confirm(
      'This will MOVE THE ROBOT.\n\n' +
      'Running: ' + human + '\n' +
      'Session: ' + CALIB.currentSession + '\n\n' +
      'Cancelable at any point from the Cancel button. OK to proceed?'
    );
    if (!yes) return;
  }
  CALIB_LOG.textContent = '';
  // Per-cmd dataset choices ride alongside `flags` in the POST body. The web
  // backend slots these into the subprocess argv after validating them
  // against an allowlist.
  const reqBody = {session: CALIB.currentSession, cmd, flags};
  let datasetSummary = '';
  if (customName) {
    reqBody.custom_name = customName;
    datasetSummary = ` (dataset=${customName})`;
  }
  if (cmd === 'chain') {
    const sel = document.getElementById('chain-handeye-select');
    if (sel) {
      reqBody.handeye = sel.value;
      datasetSummary = ` --handeye ${sel.value}`;
    }
  } else if (cmd === 'polish') {
    const phase1 = $$('#polish-phase1-checks input[type="checkbox"]:checked').map(i => i.value);
    if (phase1.length) {
      reqBody.phase1 = phase1;
      datasetSummary = ` --phase1 ${phase1.join(' ')}`;
    }
  } else if (cmd === 'validate') {
    const psel = document.getElementById('validate-params-select');
    if (psel) {
      reqBody.params = psel.value;
      datasetSummary = ` --params ${psel.value}`;
    }
  }
  const banner = cmd.startsWith('collect_')
    ? '$ ros2 run pan_tilt calibrate_collect --ros-args -p phase:=' + cmd.replace('collect_', '')
    : '$ python -m pan_tilt.calibration.run_calibration ' + cmd + datasetSummary + ' ' + flags.join(' ');
  calibLogAppend(banner, 'log-start');
  try {
    const r = await fetch('/api/calib/run', {
      method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify(reqBody),
    });
    const body = await readJsonResponse(r);
    if (!r.ok) throw new Error(body.detail || JSON.stringify(body));
    CALIB.activeRunId = body.run_id;
    CALIB_CANCEL_BTN.disabled = false;
  } catch (e) {
    calibLogAppend('ERROR: ' + e.message, 'log-exit-err');
  }
}

$$('#calib-run-buttons button[data-calib-cmd], #calib-run-buttons-collect button[data-calib-cmd]').forEach(b => {
  b.addEventListener('click', () => {
    const cmd = b.dataset.calibCmd;
    const flags = (b.dataset.calibFlags || '').split(/\s+/).filter(Boolean);
    // Solver-option toggles are default-on for the web UI and apply to the two
    // solves that fit kinematics — chain and polish. They're appended here (not
    // baked into data-calib-flags) so the checkboxes control them live.
    if (cmd === 'chain' || cmd === 'polish') {
      // Pan-axis tilt (non-vertical pan axis). Off → legacy vertical-axis model.
      const panAxisTilt = document.getElementById('chk-pan-axis-tilt');
      if (panAxisTilt && panAxisTilt.checked) flags.push('--fit-pan-axis-tilt');
      // Unlock T_B rotation. Off → lock T_B at the warm-start (chain's legacy
      // default; the degeneracy-safe fallback if a fit reports ~20° rot RMSE).
      const unlockTb = document.getElementById('chk-unlock-tb');
      if (unlockTb && unlockTb.checked) flags.push('--unlock-tb-rotation');
    }
    calibRun(cmd, flags);
  });
});

// Re-evaluate run-button enablement when the dataset selectors change. The
// underlying `files` map only refreshes on session-detail load, so we cache
// it in CALIB.lastFiles and feed it back to calibApplyRunEnablement here.
const _chainHandeyeSel = document.getElementById('chain-handeye-select');
if (_chainHandeyeSel) {
  _chainHandeyeSel.addEventListener('change', () => {
    calibApplyRunEnablement(CALIB.lastFiles || {});
  });
}
// Delegated: the polish phase-1 checkboxes are rebuilt whenever the custom
// dataset list changes, so bind on the stable container, not each input.
const _polishChecks = document.getElementById('polish-phase1-checks');
if (_polishChecks) {
  _polishChecks.addEventListener('change', () => {
    calibApplyRunEnablement(CALIB.lastFiles || {});
  });
}
const _validateParamsSel = document.getElementById('validate-params-select');
if (_validateParamsSel) {
  _validateParamsSel.addEventListener('change', () => {
    calibApplyRunEnablement(CALIB.lastFiles || {});
  });
}

CALIB_CANCEL_BTN.addEventListener('click', async () => {
  if (!CALIB.activeRunId) return;
  try {
    await fetch(`/api/calib/runs/${CALIB.activeRunId}/cancel`, {method: 'POST'});
  } catch (e) { calibLogAppend('cancel ERROR: ' + e, 'log-exit-err'); }
});

// ---- /ws/calib-log --------------------------------------------------------

function calibLogConnect() {
  connectWS('/ws/calib-log', {
    onMessage: (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === 'log') {
        calibLogAppend(msg.line);
      } else if (msg.type === 'exit') {
        const cls = msg.code === 0 ? 'log-exit-ok' : 'log-exit-err';
        calibLogAppend(`[ exited ${msg.code} ]`, cls);
        CALIB_CANCEL_BTN.disabled = true;
        CALIB.activeRunId = null;
        if (CALIB.currentSession) {
          CALIB.fileCache = {};  // session detail will repopulate
          calibLoadSessionDetail(CALIB.currentSession).catch(console.warn);
        }
      } else if (msg.type === 'start') {
        calibLogAppend(`[ started pid=${msg.pid} ]`, 'log-start');
      }
    },
  });
}

// ---- residual charts (vanilla canvas) --------------------------------------

$$('input[name="calib-resid"]').forEach(r => r.addEventListener('change', () => {
  CALIB.residualSource = document.querySelector('input[name="calib-resid"]:checked').value;
  calibRenderResiduals();
}));

$$('input[name="calib-cov"]').forEach(r => r.addEventListener('change', () => {
  calibRenderCoverage();
}));

async function calibRenderResiduals() {
  const histCanvas = $('#calib-resid-hist');
  const scatterCanvas = $('#calib-resid-scatter');
  if (!histCanvas || !scatterCanvas) return;
  _clearCanvas(histCanvas); _clearCanvas(scatterCanvas);
  if (!CALIB.currentSession) return;
  const fileMap = {handeye: 'handeye.json', chain: 'chain.json', polish: 'polish.json'};
  const filename = fileMap[CALIB.residualSource];
  const j = await calibFetchFile(CALIB.currentSession, filename);
  if (!j) return;
  const trans = j.per_sample_trans_err_m || [];
  if (trans.length === 0) {
    _ctxNote(histCanvas, '(no per-sample data)');
    _ctxNote(scatterCanvas, '(no per-sample data)');
    return;
  }
  _drawHistogram(histCanvas, trans.map(x => x * 1000), 'mm');
  _drawScatter(scatterCanvas, trans.map(x => x * 1000), 'mm');
}

async function calibRenderCoverage() {
  const c = $('#calib-coverage');
  const status = $('#calib-coverage-status');
  if (!c) return;
  _clearCanvas(c);
  if (!CALIB.currentSession) {
    if (status) status.textContent = '';
    return;
  }
  let body;
  try {
    const r = await fetch(`/api/calib/session/${encodeURIComponent(CALIB.currentSession)}/coverage`);
    if (!r.ok) { _ctxNote(c, '(no coverage data)'); return; }
    body = await readJsonResponse(r);
  } catch (e) { _ctxNote(c, '(fetch failed)'); return; }

  const allSamples = body.samples || [];
  const filterRadio = document.querySelector('input[name="calib-cov"]:checked');
  const filter = filterRadio ? filterRadio.value : 'all';
  const samples = (filter === 'all')
    ? allSamples
    : allSamples.filter(s => s.phase === filter);
  if (samples.length === 0) {
    const phaseLabel = ({
      all: '',
      phase1_handeye: 'Phase 1 (level)',
      phase1_handeye_custom: 'Phase 1 (custom)',
      phase2_chain: 'Phase 2',
    })[filter] || filter;
    _ctxNote(c, allSamples.length === 0
      ? '(no samples in this session yet — run Phase 1/2 collect)'
      : `(no samples for ${phaseLabel} — collect or switch to a populated set)`);
    if (status) status.textContent = '';
    return;
  }

  const ctx = c.getContext('2d');
  const pad = 32;
  const W = c.width - 2 * pad, H = c.height - 2 * pad;

  // Plot range: max of FoV box or sample extents (so off-axis samples are visible).
  const fovH = body.fov_h_deg / 2, fovV = body.fov_v_deg / 2;
  const sampleH = Math.max(...samples.map(s => Math.abs(s.horiz_deg)));
  const sampleV = Math.max(...samples.map(s => Math.abs(s.vert_deg)));
  const rangeH = Math.max(fovH, sampleH) * 1.1;
  const rangeV = Math.max(fovV, sampleV) * 1.1;

  const x2px = (deg) => pad + W * (deg + rangeH) / (2 * rangeH);
  const y2px = (deg) => pad + H * (deg + rangeV) / (2 * rangeV);

  // Background grid + axes
  ctx.strokeStyle = '#2c3140';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x2px(0), pad); ctx.lineTo(x2px(0), pad + H);
  ctx.moveTo(pad, y2px(0)); ctx.lineTo(pad + W, y2px(0));
  ctx.stroke();

  // FoV box
  ctx.strokeStyle = '#5fd37f';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(x2px(-fovH), y2px(-fovV), x2px(fovH) - x2px(-fovH), y2px(fovV) - y2px(-fovV));
  ctx.setLineDash([]);

  // Depth color scale across all samples
  const depths = samples.map(s => s.depth_m);
  const dMin = Math.min(...depths), dMax = Math.max(...depths);
  const dSpan = (dMax - dMin) || 1;
  const colorAt = (d) => {
    // Warm (close) -> cool (far): red -> blue
    const t = (d - dMin) / dSpan;
    const r = Math.round(255 * (1 - t));
    const b = Math.round(255 * t);
    return `rgba(${r}, 80, ${b}, 0.85)`;
  };

  // Phase shape: phase1 = filled circle, phase2 = open triangle
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  samples.forEach((s, i) => {
    const x = x2px(s.horiz_deg), y = y2px(s.vert_deg);
    const col = colorAt(s.depth_m);
    ctx.fillStyle = col;
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    if (s.phase === 'phase2_chain') {
      ctx.moveTo(x, y - 3.5); ctx.lineTo(x + 3, y + 3); ctx.lineTo(x - 3, y + 3); ctx.closePath();
      ctx.stroke();
    } else {
      ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }
    // Always-on index label so it lines up with the projected image panel.
    // Fall back to array position if the backend payload didn't include it
    // (older calib_web running, etc).
    const idx = (s.index !== undefined && s.index !== null) ? s.index : i;
    ctx.fillStyle = col;
    ctx.fillText(`#${idx}`, x + 5, y - 4);
  });

  // Axis labels
  ctx.fillStyle = '#8a93a6'; ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(`horiz angle (deg)  →  +${rangeH.toFixed(0)}`, pad + W, pad + H + 14);
  ctx.fillText(`-${rangeH.toFixed(0)}`, pad, pad + H + 14);
  ctx.textAlign = 'left';
  ctx.fillText(`vert -${rangeV.toFixed(0)}`, 2, pad + 8);
  ctx.fillText(`vert +${rangeV.toFixed(0)}`, 2, pad + H);
  ctx.textAlign = 'left';
  ctx.fillText(`FoV ${body.fov_h_deg.toFixed(0)}° × ${body.fov_v_deg.toFixed(0)}°`,
               pad, pad - 4);
  ctx.textAlign = 'right';
  ctx.fillText(`● phase1   △ phase2   depth ${dMin.toFixed(2)}-${dMax.toFixed(2)} m`,
               pad + W, pad - 4);
  if (status) {
    const n1 = samples.filter(s => s.phase === 'phase1_handeye').length;
    const n2 = samples.length - n1;
    status.textContent = `${samples.length} samples (${n1} phase1, ${n2} phase2)`;
  }
  calibRenderCoverageProjection(body, samples, dMin, dMax);
}

function calibRenderCoverageProjection(body, samples, dMin, dMax) {
  const c = document.getElementById('calib-coverage-img');
  const status = document.getElementById('calib-coverage-img-status');
  if (!c) return;
  _clearCanvas(c);

  if (!body.have_intrinsics || !body.image_w || !body.image_h) {
    _ctxNote(c, '(no camera intrinsics yet — start the camera so K is published)');
    if (status) status.textContent = '';
    return;
  }

  const ctx = c.getContext('2d');
  const imgW = body.image_w, imgH = body.image_h;
  // Letterbox the image rectangle into the canvas while keeping aspect ratio.
  const pad = 6;
  const availW = c.width - 2 * pad, availH = c.height - 2 * pad;
  const scale = Math.min(availW / imgW, availH / imgH);
  const drawW = imgW * scale, drawH = imgH * scale;
  const x0 = pad + (availW - drawW) / 2;
  const y0 = pad + (availH - drawH) / 2;

  // Monocolor background = simulated image plane.
  ctx.fillStyle = '#1b1f2a';
  ctx.fillRect(x0, y0, drawW, drawH);
  ctx.strokeStyle = '#2c3140';
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, drawW - 1, drawH - 1);

  // Same depth ramp as the angular plot.
  const dSpan = (dMax - dMin) || 1;
  const colorAt = (d) => {
    const t = (d - dMin) / dSpan;
    const r = Math.round(255 * (1 - t));
    const b = Math.round(255 * t);
    return `rgba(${r}, 80, ${b}, 0.95)`;
  };

  ctx.font = '10px monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  let nProjected = 0, nOff = 0;
  samples.forEach((s, i) => {
    if (s.u_norm == null || s.v_norm == null) return;
    const u = s.u_norm, v = s.v_norm;
    const onImage = (u >= 0 && u <= 1 && v >= 0 && v <= 1);
    if (onImage) nProjected++; else nOff++;
    // Clamp slightly outside so off-image points still render at the edge.
    const cu = Math.max(-0.05, Math.min(1.05, u));
    const cv = Math.max(-0.05, Math.min(1.05, v));
    const x = x0 + cu * drawW;
    const y = y0 + cv * drawH;
    const col = colorAt(s.depth_m);
    ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    ctx.beginPath();
    if (s.phase === 'phase2_chain') {
      ctx.moveTo(x, y - 5); ctx.lineTo(x + 4.5, y + 4); ctx.lineTo(x - 4.5, y + 4); ctx.closePath();
      ctx.stroke();
    } else {
      ctx.arc(x, y, 4.5, 0, 2 * Math.PI);
      ctx.fill();
    }
    const idx = (s.index !== undefined && s.index !== null) ? s.index : i;
    ctx.fillStyle = col;
    ctx.fillText(`#${idx}`, x + 6, y - 5);
  });

  // Caption: image dims + on/off counts.
  ctx.fillStyle = '#8a93a6';
  ctx.font = '10px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(`image ${imgW}×${imgH}`, x0, y0 - 2);
  ctx.textAlign = 'right';
  ctx.fillText(`on-image ${nProjected}   off ${nOff}`, x0 + drawW, y0 - 2);

  if (status) status.textContent = '';
}

function _clearCanvas(c) {
  const ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height);
}
function _ctxNote(c, text) {
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#8a93a6';
  ctx.font = '12px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(text, c.width / 2, c.height / 2);
}
function _drawHistogram(c, values, unit) {
  const ctx = c.getContext('2d');
  const pad = 24;
  const W = c.width - 2 * pad, H = c.height - 2 * pad;
  const nBins = 20;
  const lo = Math.min(...values), hi = Math.max(...values);
  const span = hi - lo || 1;
  const bins = Array(nBins).fill(0);
  values.forEach(v => {
    const i = Math.min(nBins - 1, Math.floor((v - lo) / span * nBins));
    bins[i]++;
  });
  const maxBin = Math.max(...bins);
  // Axes
  ctx.strokeStyle = '#2c3140'; ctx.beginPath();
  ctx.moveTo(pad, pad); ctx.lineTo(pad, pad + H); ctx.lineTo(pad + W, pad + H); ctx.stroke();
  // Bars
  const bw = W / nBins;
  ctx.fillStyle = '#4da3ff';
  bins.forEach((n, i) => {
    const h = maxBin > 0 ? (n / maxBin) * H : 0;
    ctx.fillRect(pad + i * bw + 1, pad + H - h, bw - 2, h);
  });
  // Labels
  ctx.fillStyle = '#8a93a6'; ctx.font = '10px monospace'; ctx.textAlign = 'left';
  ctx.fillText(`${lo.toFixed(2)} ${unit}`, pad, pad + H + 12);
  ctx.textAlign = 'right';
  ctx.fillText(`${hi.toFixed(2)} ${unit}`, pad + W, pad + H + 12);
  ctx.textAlign = 'left';
  ctx.fillText(`N=${values.length}  max=${maxBin}`, pad, pad - 4);
}
function _drawScatter(c, values, unit) {
  const ctx = c.getContext('2d');
  const pad = 24;
  const W = c.width - 2 * pad, H = c.height - 2 * pad;
  const lo = 0, hi = Math.max(...values) * 1.05 || 1;
  // Axes
  ctx.strokeStyle = '#2c3140'; ctx.beginPath();
  ctx.moveTo(pad, pad); ctx.lineTo(pad, pad + H); ctx.lineTo(pad + W, pad + H); ctx.stroke();
  // Points
  ctx.fillStyle = '#5fd37f';
  const n = values.length;
  values.forEach((v, i) => {
    const x = pad + (n > 1 ? (i / (n - 1)) * W : W / 2);
    const y = pad + H - (v - lo) / (hi - lo) * H;
    ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 2 * Math.PI); ctx.fill();
  });
  // Labels
  ctx.fillStyle = '#8a93a6'; ctx.font = '10px monospace'; ctx.textAlign = 'left';
  ctx.fillText(`0 ${unit}`, 2, pad + H + 12);
  ctx.fillText(`sample 0`, pad, pad + H + 12);
  ctx.textAlign = 'right';
  ctx.fillText(`${hi.toFixed(2)} ${unit}`, pad + W, pad - 4);
  ctx.fillText(`sample ${n - 1}`, pad + W, pad + H + 12);
}

// ---- URDF diff -------------------------------------------------------------

async function calibLoadUrdfTargets() {
  try {
    const r = await fetch('/api/calib/urdf_targets');
    const body = await readJsonResponse(r);
    CALIB.urdfTargets = body.targets || [];
    const sel = $('#calib-urdf-target');
    sel.innerHTML = CALIB.urdfTargets
      .map(t => `<option value="${t.path}" ${t.exists ? '' : 'disabled'}>${t.label}${t.exists ? '' : '  (not installed)'}</option>`)
      .join('');
  } catch (e) {
    $('#calib-urdf-status').textContent = 'ERROR loading URDF targets: ' + e;
    $('#calib-urdf-status').className = 'status-line err';
  }
}

$('#calib-urdf-diff-btn').addEventListener('click', async () => {
  if (!CALIB.currentSession) {
    $('#calib-urdf-status').textContent = 'pick a session first';
    $('#calib-urdf-status').className = 'status-line warn';
    return;
  }
  const xacroPath = $('#calib-urdf-target').value;
  const resultsFile = $('#calib-urdf-results').value;
  const statusEl = $('#calib-urdf-status');
  const diffEl = $('#calib-urdf-diff');
  statusEl.textContent = 'generating…';
  statusEl.className = 'status-line warn';
  try {
    const r = await fetch('/api/calib/urdf_diff', {
      method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify({session: CALIB.currentSession, xacro_path: xacroPath, results_file: resultsFile}),
    });
    const body = await readJsonResponse(r);
    if (!r.ok) throw new Error(body.detail || JSON.stringify(body));
    const urdfDiff = body.diff || '(no URDF changes)';
    const yamlDiff = body.yaml_diff || '';
    const yamlPath = body.yaml_path || '';
    let combined = urdfDiff;
    if (yamlDiff) {
      combined += '\n\n' + (
        yamlPath
          ? `--- ${yamlPath} (runtime offsets)\n+++ ${yamlPath} (calibrated)\n`
          : ''
      ) + yamlDiff;
    } else if (yamlPath) {
      combined += `\n\n# YAML ${yamlPath} already matches calibration\n`;
    }
    diffEl.innerHTML = _colorizeDiff(combined);
    const changed = (urdfDiff + '\n' + yamlDiff)
      .split('\n').filter(l => l.startsWith('+') || l.startsWith('-')).length;
    statusEl.textContent = `diff: ${changed} changed lines (URDF + YAML)`;
    statusEl.className = 'status-line ok';
  } catch (e) {
    statusEl.textContent = 'FAIL: ' + e.message;
    statusEl.className = 'status-line err';
    diffEl.textContent = '';
  }
});

function _colorizeDiff(diff) {
  return diff.split('\n').map(line => {
    const esc = line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    if (line.startsWith('+++') || line.startsWith('---')) return `<span class="diff-hunk">${esc}</span>`;
    if (line.startsWith('@@')) return `<span class="diff-hunk">${esc}</span>`;
    if (line.startsWith('+')) return `<span class="diff-add">${esc}</span>`;
    if (line.startsWith('-')) return `<span class="diff-rem">${esc}</span>`;
    return esc;
  }).join('\n');
}

$('#calib-urdf-apply-btn').addEventListener('click', async () => {
  if (!CALIB.currentSession) {
    $('#calib-urdf-status').textContent = 'pick a session first';
    $('#calib-urdf-status').className = 'status-line warn';
    return;
  }
  const xacroPath = $('#calib-urdf-target').value;
  if (!xacroPath) {
    $('#calib-urdf-status').textContent = 'pick a URDF target first';
    $('#calib-urdf-status').className = 'status-line warn';
    return;
  }
  const target = CALIB.urdfTargets.find(t => t.path === xacroPath);
  const resultsFile = $('#calib-urdf-results').value || 'polish.json';
  const statusEl = $('#calib-urdf-status');
  const rebuildEl = $('#calib-urdf-rebuild');
  rebuildEl.hidden = true;
  statusEl.textContent = 'applying…';
  statusEl.className = 'status-line warn';
  try {
    const r = await fetch('/api/calib/urdf_apply', {
      method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify({session: CALIB.currentSession, xacro_path: xacroPath, results_file: resultsFile}),
    });
    const body = await readJsonResponse(r);
    if (!r.ok) throw new Error(body.detail || JSON.stringify(body));
    if (!body.applied) {
      statusEl.textContent = body.reason || 'no change — xacro already matches calibration';
      statusEl.className = 'status-line ok';
      return;
    }
    statusEl.textContent = `patched ${target?.label || xacroPath} — rebuild required`;
    statusEl.className = 'status-line ok';
    $('#calib-urdf-rebuild-cmd').textContent = body.build_command || '';
    $('#calib-urdf-rebuild-hint').textContent = body.workspace_hint || '';
    // Two-line backup hint: URDF + YAML. The YAML line surfaces whether
    // the runtime offsets just changed (or were already current) so the
    // operator never wonders whether they still need to hand-edit it.
    const backupLines = [];
    if (body.backup_path) {
      backupLines.push(`URDF backup: ${body.backup_path}`);
    }
    if (body.yaml_path) {
      if (body.yaml_applied && body.yaml_backup_path) {
        const pan = body.pan_offset_rad?.toFixed(10) ?? '?';
        const tilt = body.tilt_offset_rad?.toFixed(10) ?? '?';
        backupLines.push(
          `YAML backup: ${body.yaml_backup_path}`,
          `Runtime offsets: pan=${pan}, tilt=${tilt}`,
        );
      } else {
        backupLines.push(`YAML: already matches calibration (no change)`);
      }
    }
    $('#calib-urdf-backup-hint').textContent = backupLines.join('\n');
    rebuildEl.hidden = false;
    // Combined diff preview: URDF first, YAML below.
    let preview = body.diff_preview || '';
    if (body.yaml_diff_preview) {
      preview += (preview ? '\n\n' : '') + body.yaml_diff_preview;
    }
    if (preview) {
      $('#calib-urdf-diff').innerHTML = _colorizeDiff(preview);
    }
  } catch (e) {
    statusEl.textContent = 'FAIL: ' + e.message;
    statusEl.className = 'status-line err';
  }
});

// ---- run-button enablement (prereq check + collect-availability) ----------

async function calibLoadCommands() {
  try {
    const r = await fetch('/api/calib/commands');
    const body = await readJsonResponse(r);
    if (!r.ok) throw new Error(body.detail || 'commands fetch failed');
    CALIB.prereqs = body.prereqs || {};
    CALIB.collectEnabled = !!body.collect_enabled;
  } catch (e) {
    console.warn('could not load calib commands:', e);
  }
  calibApplyRunEnablement({});
}

function calibApplyRunEnablement(files) {
  const btns = $$('#calib-run-buttons button[data-calib-cmd], #calib-run-buttons-collect button[data-calib-cmd]');
  btns.forEach(b => {
    const cmd = b.dataset.calibCmd;
    if (!cmd) return;
    if (cmd.startsWith('collect_') && !CALIB.collectEnabled) {
      b.disabled = true;
      b.title = 'calibrate_web was launched without -p config:=... ; collect is unavailable';
      return;
    }
    const prereqs = CALIB.prereqs[cmd] || [];
    const missing = prereqs.filter(f => !((files[f] || {}).exists));
    // Dynamic prereqs: chain and polish accept per-request input choices, so
    // their enablement depends on current selector state in addition to the
    // static prereq list. Re-run on every dropdown / checkbox change.
    const dynMissing = calibDynamicPrereqs(cmd, files);
    const allMissing = missing.concat(dynMissing);
    if (allMissing.length > 0) {
      b.disabled = true;
      b.title = 'needs ' + allMissing.join(', ') + ' in the session first';
    } else {
      b.disabled = false;
      b.title = '';
    }
  });
}

function calibDynamicPrereqs(cmd, files) {
  if (cmd === 'chain') {
    const sel = $('#chain-handeye-select');
    const chosen = sel ? sel.value : 'handeye.json';
    return ((files[chosen] || {}).exists) ? [] : [chosen];
  }
  if (cmd === 'polish') {
    const phase1 = $$('#polish-phase1-checks input[type="checkbox"]:checked')
      .map(i => i.value);
    const missing = [];
    if (phase1.length === 0) missing.push('(at least one phase-1 dataset)');
    for (const f of phase1) {
      if (!((files[f] || {}).exists)) missing.push(f);
    }
    if (!((files['chain.json'] || {}).exists)) missing.push('chain.json');
    return missing;
  }
  if (cmd === 'validate') {
    const psel = $('#validate-params-select');
    const chosen = psel ? psel.value : 'polish.json';
    return ((files[chosen] || {}).exists) ? [] : [chosen];
  }
  return [];
}

// JS mirror of pan_tilt.calibration.custom_naming.custom_dataset_filenames.
function customSolveFile(name) {
  return name === 'custom' ? 'handeye_custom.json' : `handeye_custom_${name}.json`;
}
function customPhase1File(name) {
  return name === 'custom' ? 'phase1_handeye_custom.json' : `phase1_handeye_custom_${name}.json`;
}

// Fill the Calibrate-tab dataset picker shared by collect_phase1_custom +
// handeye_custom.
function populateCustomDatasetPicker() {
  const sel = document.getElementById('calib-custom-dataset-select');
  if (!sel) return;
  const prev = sel.value;
  sel.innerHTML = '';
  if (!customDatasets.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '(none — add in xArm Waypoints tab)';
    sel.appendChild(opt);
    return;
  }
  customDatasets.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d.name;
    opt.textContent = `${d.name} (pan=${d.park_pan_deg}°, tilt=${d.park_tilt_deg}°)`;
    sel.appendChild(opt);
  });
  if (customDatasets.some(d => d.name === prev)) sel.value = prev;
  populatePrunePhaseOptions();
}

// Inject one `phase1_waypoints_custom:<name>` option per dataset into the Prune
// tab's phase selector (between phase1_waypoints and phase2_grid).
function populatePrunePhaseOptions() {
  const sel = document.getElementById('prune-phase');
  if (!sel) return;
  const prev = sel.value;
  // Drop any previously-injected custom options.
  Array.from(sel.options)
    .filter(o => o.value.startsWith(CUSTOM_PHASE_PREFIX))
    .forEach(o => o.remove());
  const grid = Array.from(sel.options).find(o => o.value === 'phase2_grid');
  customDatasets.forEach(d => {
    const opt = document.createElement('option');
    opt.value = _customKey(d.name);
    opt.textContent = `phase1_waypoints_custom: ${d.name}`;
    sel.insertBefore(opt, grid || null);
  });
  if (Array.from(sel.options).some(o => o.value === prev)) sel.value = prev;
}

// Rebuild the chain `--handeye` dropdown + polish `--phase1` checkboxes so they
// list the canonical solve plus every custom dataset's solve/phase1 file.
function rebuildHandeyeSelectors() {
  const sel = document.getElementById('chain-handeye-select');
  if (sel) {
    const prev = sel.value;
    sel.innerHTML = '';
    const opts = [['handeye.json', 'handeye.json (level)']];
    customDatasets.forEach(d =>
      opts.push([customSolveFile(d.name), `${customSolveFile(d.name)} (${d.name})`]));
    opts.forEach(([val, label]) => {
      const o = document.createElement('option');
      o.value = val; o.textContent = label; sel.appendChild(o);
    });
    sel.value = opts.some(([v]) => v === prev) ? prev : 'handeye.json';
  }
  const checks = document.getElementById('polish-phase1-checks');
  if (checks) {
    const prevChecked = new Set(
      $$('#polish-phase1-checks input[type="checkbox"]:checked').map(i => i.value));
    const firstBuild = checks.dataset.built !== '1';
    checks.innerHTML = '';
    const items = [['phase1_handeye.json', 'level']];
    customDatasets.forEach(d => items.push([customPhase1File(d.name), d.name]));
    items.forEach(([val, label]) => {
      const lbl = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox'; cb.value = val;
      cb.checked = firstBuild ? (val === 'phase1_handeye.json') : prevChecked.has(val);
      lbl.appendChild(cb); lbl.append(' ' + label);
      checks.appendChild(lbl);
    });
    checks.dataset.built = '1';
  }
  if (CALIB.lastFiles) {
    calibRenderDatasetSelectors(CALIB.lastFiles);
    calibApplyRunEnablement(CALIB.lastFiles);
  }
}

function calibRenderDatasetSelectors(files) {
  // Disable / un-disable each option based on whether its file exists in the
  // current session. Don't auto-rewrite checked state for the dropdown — keep
  // operator's last selection if still valid; otherwise fall back to first
  // enabled option. Also append a relative-time badge so an operator who
  // forgot to re-collect sees "5h ago" next to a stale option.
  const sel = $('#chain-handeye-select');
  if (sel) {
    let firstEnabled = null;
    Array.from(sel.options).forEach(opt => {
      const info = files[opt.value] || {};
      const exists = !!info.exists;
      opt.disabled = !exists;
      // Cache original label once so re-renders don't accumulate badges.
      if (!opt.dataset.baseLabel) opt.dataset.baseLabel = opt.text;
      const stamp = exists ? _calibRelTime(info.mtime) : 'missing';
      opt.text = stamp ? `${opt.dataset.baseLabel} — ${stamp}` : opt.dataset.baseLabel;
      if (exists && firstEnabled === null) firstEnabled = opt.value;
    });
    if (sel.options[sel.selectedIndex]?.disabled && firstEnabled) {
      sel.value = firstEnabled;
    }
  }
  const checks = $$('#polish-phase1-checks input[type="checkbox"]');
  let anyChecked = false;
  checks.forEach(cb => {
    const info = files[cb.value] || {};
    const exists = !!info.exists;
    cb.disabled = !exists;
    if (cb.checked && !exists) cb.checked = false;
    if (cb.checked) anyChecked = true;
    // Replace the text node that follows the checkbox with "<base> — <when>".
    const lbl = cb.parentElement;
    if (lbl) {
      if (!lbl.dataset.baseLabel) {
        // First render — capture the original label content (text after the input).
        const baseText = (lbl.textContent || '').trim();
        lbl.dataset.baseLabel = baseText;
      }
      const stamp = exists ? _calibRelTime(info.mtime) : 'missing';
      // Rebuild: keep the input element, drop other children, add fresh text.
      while (lbl.childNodes.length > 1) lbl.removeChild(lbl.lastChild);
      lbl.appendChild(document.createTextNode(
        stamp ? ` ${lbl.dataset.baseLabel} — ${stamp}` : ` ${lbl.dataset.baseLabel}`
      ));
    }
  });
  // If nothing is checked but a dataset is available, default-check the first.
  if (!anyChecked) {
    const firstAvailable = checks.find(cb => !cb.disabled);
    if (firstAvailable) firstAvailable.checked = true;
  }

  // Validate params dropdown: same pattern as the chain handeye dropdown —
  // disable missing options and show staleness inline.
  const vparams = $('#validate-params-select');
  if (vparams) {
    let firstEnabled = null;
    Array.from(vparams.options).forEach(opt => {
      const info = files[opt.value] || {};
      const exists = !!info.exists;
      opt.disabled = !exists;
      if (!opt.dataset.baseLabel) opt.dataset.baseLabel = opt.text;
      const stamp = exists ? _calibRelTime(info.mtime) : 'missing';
      opt.text = stamp ? `${opt.dataset.baseLabel} — ${stamp}` : opt.dataset.baseLabel;
      if (exists && firstEnabled === null) firstEnabled = opt.value;
    });
    if (vparams.options[vparams.selectedIndex]?.disabled && firstEnabled) {
      vparams.value = firstEnabled;
    }
  }
}

// ---- prune (preview-then-apply) -------------------------------------------
//
// Operator-driven waypoint pruning. Preview returns a kept/dropped headline
// and a per-waypoint table without touching disk; Apply re-runs the same
// deterministic prune (with whatever overrides the operator toggled) and
// writes a sidecar yaml. The Apply button stays disabled until a Preview
// has succeeded so the operator always sees the count first.

const PRUNE = {
  lastPayload: null,         // last request body sent to /preview
  lastResponse: null,        // last response body
  overrides: {},             // {idx: 'keep'|'drop'} per session
  defaultsByPhase: {},       // populated lazily from /api/calib/prune_inputs
};

function pruneCurrentFactors() {
  return {
    trans_tol_m: parseFloat($('#prune-trans-tol').value),
    rot_tol_deg: parseFloat($('#prune-rot-tol').value),
    min_count: parseInt($('#prune-min-count').value, 10),
    min_rot_diversity_pairs: parseInt($('#prune-min-rot-pairs').value, 10),
    min_rot_diversity_deg: parseFloat($('#prune-min-rot-deg').value),
    seed_index: parseInt($('#prune-seed').value, 10),
  };
}

function prunePopulateFactors(factors) {
  $('#prune-trans-tol').value = factors.trans_tol_m;
  $('#prune-rot-tol').value = factors.rot_tol_deg;
  $('#prune-min-count').value = factors.min_count;
  $('#prune-min-rot-pairs').value = factors.min_rot_diversity_pairs;
  $('#prune-min-rot-deg').value = factors.min_rot_diversity_deg;
  $('#prune-seed').value = factors.seed_index;
}

async function pruneLoadInputs() {
  const phase = $('#prune-phase').value;
  try {
    const r = await fetch(`/api/calib/prune_inputs?phase=${encodeURIComponent(phase)}`);
    if (!r.ok) throw new Error(await r.text());
    const data = await readJsonResponse(r);
    PRUNE.defaultsByPhase[phase] = data.default_factors;
    prunePopulateFactors(data.default_factors);

    // Prior-run picker.
    const sel = $('#prune-prior-run');
    sel.innerHTML = '<option value="">(none — fk only)</option>';
    for (const run of data.prior_runs) {
      const opt = document.createElement('option');
      opt.value = run.path;
      opt.textContent = `${run.name} (${run.n_samples ?? '?'} samples)`;
      sel.appendChild(opt);
    }
    if (data.prior_runs.length > 0) sel.value = data.prior_runs[0].path;

    $('#prune-status').textContent = `loaded ${data.n_items} items for phase=${phase}`;
    // Reset apply/overwrite state on phase change.
    PRUNE.lastResponse = null;
    PRUNE.overrides = {};
    $('#prune-apply-btn').disabled = true;
    $('#prune-overwrite-btn').disabled = true;
    $('#prune-table').hidden = true;
    $('#prune-tbody').innerHTML = '';
    $('#prune-headline').textContent = '';
    $('#prune-warning').hidden = true;
    $('#prune-diagnostics').hidden = true;
  } catch (e) {
    $('#prune-status').textContent = `failed to load inputs: ${e}`;
  }
}

// Resolve the source-tree promote target so the Prune tab shows the
// operator exactly which file Overwrite would replace and what the backup
// will be named. Reuses /api/waypoints/paths from the existing waypoint
// flow — single source of truth.
async function pruneLoadPaths() {
  const src = $('#prune-path-source');
  const bak = $('#prune-path-backup');
  if (!src || !bak) return;
  try {
    const r = await fetch('/api/waypoints/paths');
    if (!r.ok) throw new Error(await r.text());
    const data = await readJsonResponse(r);
    if (data.promote) {
      src.textContent = data.promote;
      bak.textContent = data.promote.replace(/\.yaml$/, '.yaml.old-<YYYYmmdd_HHMMSS>');
      PRUNE.promoteTarget = data.promote;
    } else {
      src.textContent = '(none — Overwrite disabled)';
      bak.textContent = '–';
      PRUNE.promoteTarget = null;
    }
  } catch (e) {
    src.textContent = `failed: ${e}`;
    bak.textContent = '–';
    PRUNE.promoteTarget = null;
  }
}

function pruneBuildPayload(extra) {
  return {
    phase: $('#prune-phase').value,
    factors: pruneCurrentFactors(),
    predictor_choice: $('#prune-predictor').value,
    prior_run_path: $('#prune-prior-run').value || null,
    overrides: { ...PRUNE.overrides },
    ...extra,
  };
}

async function prunePreview() {
  const body = pruneBuildPayload();
  $('#prune-status').textContent = 'previewing…';
  try {
    const r = await fetch('/api/calib/prune_preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await readJsonResponse(r);
    PRUNE.lastPayload = body;
    PRUNE.lastResponse = data;
    $('#prune-status').textContent = 'preview ready — Apply (sidecar) or Overwrite (replace source yaml)';
    $('#prune-headline').textContent = data.headline || '';
    $('#prune-apply-btn').disabled = false;
    $('#prune-overwrite-btn').disabled = !PRUNE.promoteTarget;
    pruneRenderDiagnostics(data);
    pruneRenderTable(data);
  } catch (e) {
    PRUNE.lastResponse = null;
    $('#prune-status').textContent = `preview failed: ${e}`;
    $('#prune-apply-btn').disabled = true;
    $('#prune-overwrite-btn').disabled = true;
  }
}

async function pruneApply() {
  if (!PRUNE.lastResponse) {
    $('#prune-status').textContent = 'click Preview first';
    return;
  }
  const headline = PRUNE.lastResponse.headline || '(no headline)';
  if (!confirm(`Write sidecar yaml?\n\n${headline}\n\n` +
               `Original calibration.yaml is untouched.`)) {
    return;
  }
  const body = pruneBuildPayload({ confirm: true });
  $('#prune-status').textContent = 'applying…';
  try {
    const r = await fetch('/api/calib/prune_apply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await readJsonResponse(r);
    PRUNE.lastResponse = data;
    pruneRenderDiagnostics(data);
    pruneRenderTable(data);
    if (data.wrote && data.wrote.sidecar_yaml) {
      $('#prune-status').textContent =
        `wrote ${data.wrote.sidecar_yaml} (${headline})`;
    } else {
      $('#prune-status').textContent = `apply succeeded: ${headline}`;
    }
    // Disable both write buttons after a successful write to prevent double-apply.
    $('#prune-apply-btn').disabled = true;
    $('#prune-overwrite-btn').disabled = true;
  } catch (e) {
    $('#prune-status').textContent = `apply failed: ${e}`;
  }
}

async function pruneOverwrite() {
  if (!PRUNE.lastResponse) {
    $('#prune-status').textContent = 'click Preview first';
    return;
  }
  if (!PRUNE.promoteTarget) {
    $('#prune-status').textContent =
      'no source-tree calibration.yaml resolved — Overwrite unavailable';
    return;
  }
  const headline = PRUNE.lastResponse.headline || '(no headline)';
  const target = PRUNE.promoteTarget;
  if (!confirm(
    `Replace ${target} with the pruned set?\n\n${headline}\n\n` +
    `Current file will be renamed to <stem>.yaml.old-<timestamp> ` +
    `in the same directory before the new yaml is written.`,
  )) {
    return;
  }
  const body = pruneBuildPayload({ confirm: true });
  $('#prune-status').textContent = 'overwriting…';
  try {
    const r = await fetch('/api/calib/prune_overwrite', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await readJsonResponse(r);
    PRUNE.lastResponse = data;
    pruneRenderDiagnostics(data);
    pruneRenderTable(data);
    const w = data.wrote || {};
    if (w.wrote_yaml) {
      const bak = w.backup_yaml ? ` · backup ${w.backup_yaml}` : '';
      $('#prune-status').textContent = `wrote ${w.wrote_yaml}${bak} (${headline})`;
    } else {
      $('#prune-status').textContent = `overwrite succeeded: ${headline}`;
    }
    $('#prune-apply-btn').disabled = true;
    $('#prune-overwrite-btn').disabled = true;
  } catch (e) {
    $('#prune-status').textContent = `overwrite failed: ${e}`;
  }
}

function pruneRenderDiagnostics(data) {
  const tbl = $('#prune-diagnostics');
  tbl.innerHTML = '';
  tbl.hidden = false;
  const diag = data.diagnostics || {};
  const rows = [
    ['kept', `${data.kept_indices.length} of ${data.items.length}`],
    ['rotation-diverse pairs in kept', diag.rot_diverse_pairs_in_kept ?? '–'],
    ['rescued for rotation diversity', diag.n_rescued_for_rot_diversity ?? 0],
    ['predict failed', diag.n_predict_failed ?? 0],
    ['forced keep / drop', `${diag.n_forced_keep ?? 0} / ${diag.n_forced_drop ?? 0}`],
    ['predictor sources', JSON.stringify(diag.predictor_sources ?? {})],
  ];
  for (const [k, v] of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${k}</td><td>${v}</td>`;
    tbl.appendChild(tr);
  }
  const warn = $('#prune-warning');
  if (diag.warning) {
    warn.textContent = diag.warning;
    warn.hidden = false;
  } else {
    warn.hidden = true;
  }
}

function pruneRenderTable(data) {
  const tbody = $('#prune-tbody');
  tbody.innerHTML = '';
  for (const it of data.items) {
    const tr = document.createElement('tr');
    tr.classList.add(it.kept ? 'prune-kept' : 'prune-dropped');
    if (it.forced_keep) tr.classList.add('prune-forced-keep');
    if (it.forced_drop) tr.classList.add('prune-forced-drop');

    const trans = it.nearest_trans_m == null ? '–' : `${(it.nearest_trans_m * 1000).toFixed(1)} mm`;
    const rot = it.nearest_rot_deg == null ? '–' : `${it.nearest_rot_deg.toFixed(2)} °`;
    const reason = it.kept ? '' : (it.drop_reason || 'dropped');
    tr.innerHTML = `
      <td>${it.index}</td>
      <td><code>${it.label}</code></td>
      <td>${it.kept ? '✔' : '✗'}</td>
      <td>${it.predictor_source ?? ''}</td>
      <td>${it.nearest_kept_label ?? ''}</td>
      <td>${trans}</td>
      <td>${rot}</td>
      <td>${reason}</td>
      <td>
        <select data-prune-override="${it.index}">
          <option value="">auto</option>
          <option value="keep">force keep</option>
          <option value="drop">force drop</option>
        </select>
      </td>
    `;
    const sel = tr.querySelector('select');
    sel.value = PRUNE.overrides[it.index] || '';
    sel.addEventListener('change', () => {
      if (sel.value) {
        PRUNE.overrides[it.index] = sel.value;
      } else {
        delete PRUNE.overrides[it.index];
      }
      // After any override edit, force a re-Preview so the write buttons
      // reflect the new override.
      $('#prune-apply-btn').disabled = true;
      $('#prune-overwrite-btn').disabled = true;
      $('#prune-status').textContent = 'overrides changed — click Preview to refresh';
    });
    tbody.appendChild(tr);
  }
  $('#prune-table').hidden = false;
}

function pruneResetFactors() {
  const phase = $('#prune-phase').value;
  const defaults = PRUNE.defaultsByPhase[phase];
  if (!defaults) {
    pruneLoadInputs();
    return;
  }
  prunePopulateFactors(defaults);
  PRUNE.overrides = {};
  $('#prune-status').textContent = 'factors reset';
  $('#prune-apply-btn').disabled = true;
  $('#prune-overwrite-btn').disabled = true;
}

if ($('#prune-phase')) {
  $('#prune-phase').addEventListener('change', pruneLoadInputs);
  $('#prune-preview-btn').addEventListener('click', prunePreview);
  $('#prune-apply-btn').addEventListener('click', pruneApply);
  $('#prune-overwrite-btn').addEventListener('click', pruneOverwrite);
  $('#prune-reset-btn').addEventListener('click', pruneResetFactors);
  // Initial load — populates the prior-run picker and the default factors.
  pruneLoadInputs().catch(e => console.warn('prune-init:', e));
  pruneLoadPaths().catch(e => console.warn('prune-paths:', e));
}

// ---- boot ------------------------------------------------------------------

// Three independent fetches + the WS connect can all start in parallel; the
// commands fetch just needs to land before the session-detail render fires
// `calibApplyRunEnablement`. Loading sessions internally awaits commands
// where needed via the next render tick (commands fill into CALIB.prereqs
// asynchronously; if sessions render first the buttons stay disabled until
// commands lands, then a final calibApplyRunEnablement re-runs).
Promise.all([calibLoadCommands(), calibLoadSessions(), calibLoadUrdfTargets()])
  .catch(e => console.warn('calibrate-tab boot:', e));
calibLogConnect();
