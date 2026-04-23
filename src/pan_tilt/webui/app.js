// calibrate_web frontend. Vanilla JS, no build step.

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

// ---- tabs -------------------------------------------------------------------

function activateTab(name) {
  $$('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  $$('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
}
$$('.tab').forEach(b => b.addEventListener('click', () => activateTab(b.dataset.tab)));

// ---- camera refresh (MJPEG-ish polling) -------------------------------------
// Re-fetch /api/frame.jpg with cache-busting at ~3 Hz. Two <img> targets
// keep the Live View + Pan-tilt tabs in sync.
// `view-mode` radios toggle between the annotated (detection overlay) stream
// and the raw camera feed — useful when overlay encoding fails (e.g. missing
// camera_info) or when debugging why frames are black.
const IMG_MAIN = $('#camera-img');
const IMG_ALT  = $('#camera-img-2');
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
  const url = currentFrameUrl();
  if (IMG_MAIN) IMG_MAIN.src = url;
  if (IMG_ALT)  IMG_ALT.src  = url;
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
      const body = await r.json();
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
let ws = null;
let lastState = null;

function wsConnect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');
  ws.onopen = () => {
    INDICATOR.textContent = 'WS: live';
    INDICATOR.classList.remove('dropped');
    INDICATOR.classList.add('connected');
  };
  ws.onclose = () => {
    INDICATOR.textContent = 'WS: dropped — retrying';
    INDICATOR.classList.remove('connected');
    INDICATOR.classList.add('dropped');
    setTimeout(wsConnect, 1500);
  };
  ws.onerror = () => {/* onclose will fire */};
  ws.onmessage = (ev) => {
    try { lastState = JSON.parse(ev.data); } catch (e) { return; }
    renderState(lastState);
  };
}
wsConnect();

// ---- state rendering --------------------------------------------------------

function fmt(v, n = 4) {
  return (v === undefined || v === null) ? '–' : Number(v).toFixed(n);
}

function renderState(s) {
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

const N_JOINTS = 6;
const JOINT_INPUTS_EL = $('#joint-inputs');
const jointInputs = [];
for (let i = 0; i < N_JOINTS; i++) {
  const row = document.createElement('div');
  row.className = 'joint-row';
  row.innerHTML = `<label>J${i + 1}</label><input type="number" step="0.01" value="0">`;
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
    const body = await r.json();
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
    const body = await r.json();
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
  { key: 'phase1_waypoints', label: 'Phase 1 — hand-eye (at pan=0,tilt=0)' },
  { key: 'phase2_waypoints', label: 'Phase 2 — anchor poses for grid sweep' },
  { key: 'sanity_xarm_angles_rad', label: 'Sanity pose (single)' },
];

const WP_ROOT = $('#waypoint-lists');
let wpState = { phase1_waypoints: [], phase2_waypoints: [], sanity_xarm_angles_rad: [] };

function renderWaypoints() {
  WP_ROOT.innerHTML = '';
  PHASES.forEach(phase => {
    const group = document.createElement('div');
    group.className = 'wp-group';
    const header = document.createElement('div');
    header.className = 'wp-header';
    header.innerHTML = `<strong>${phase.label}</strong>`;
    const add = document.createElement('button');
    add.textContent = '+ add current joints';
    add.addEventListener('click', () => addWaypoint(phase.key));
    header.appendChild(add);
    group.appendChild(header);

    const list = document.createElement('div');
    list.className = 'wp-list';

    const wps = wpState[phase.key] || [];
    const items = (phase.key === 'sanity_xarm_angles_rad')
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
      del.addEventListener('click', () => removeWaypoint(phase.key, i));
      controls.appendChild(load);
      controls.appendChild(del);
      row.appendChild(controls);
      list.appendChild(row);
    });

    group.appendChild(list);
    WP_ROOT.appendChild(group);
  });
}

async function fetchWaypoints() {
  try {
    const r = await fetch('/api/waypoints');
    if (r.ok) {
      wpState = await r.json();
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
    const body = await r.json();
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
    const body = await r.json();
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
