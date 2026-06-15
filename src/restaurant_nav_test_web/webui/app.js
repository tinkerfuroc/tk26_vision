"use strict";
const $ = (id) => document.getElementById(id);
const PROCS = ["camera_femto", "pan_tilt", "waving", "nav_driver", "nav2", "approach"];
const READY = ["camera", "pan_tilt", "waving", "goto"];

function log(msg) {
  const li = document.createElement("li");
  li.innerHTML = `<b>${new Date().toLocaleTimeString()}</b> ${msg}`;
  $("log").prepend(li);
  while ($("log").children.length > 50) $("log").lastChild.remove();
}

async function post(url) {
  try { return await (await fetch(url, { method: "POST" })).json(); }
  catch (e) { return { message: `request failed: ${e}` }; }
}

function renderState(s) {
  $("s-phase").textContent = s.phase ?? "—";
  $("s-wavers").textContent = s.waver_count ?? "—";
  $("s-target").textContent = s.target ? `(${s.target.x.toFixed(2)}, ${s.target.y.toFixed(2)})` : "—";
  $("s-result").textContent = s.result ?? "—";
  $("s-distance").textContent = (s.distance_m != null) ? `${s.distance_m} m` : "—";
  drawOverlay(s);
}

function renderProc(data) {
  const proc = data.proc || {};
  $("proc-list").innerHTML = "";
  for (const name of PROCS) {
    const st = proc[name] || { running: false };
    const li = document.createElement("li");
    const dot = st.running ? "●" : "○";
    li.innerHTML = `<span class="${st.running ? 'on' : 'off'}">${dot}</span> ${name}
      <button data-proc="${name}" data-act="${st.running ? 'stop' : 'start'}">${st.running ? 'stop' : 'start'}</button>`;
    $("proc-list").appendChild(li);
  }
  $("proc-list").querySelectorAll("button").forEach((b) => {
    b.onclick = async () =>
      log(`${b.dataset.proc} ${b.dataset.act} → ` +
          JSON.stringify(await post(`/api/proc/${b.dataset.proc}/${b.dataset.act}`)));
  });
  const r = data.readiness || {};
  $("readiness").innerHTML = READY.map(
    (k) => `<span class="${r[k] ? 'on' : 'off'}">${r[k] ? '●' : '○'} ${k}</span>`).join("  ");
}

function drawOverlay(s) {
  const img = $("video"), cv = $("overlay");
  if (!img.naturalWidth) return;
  cv.width = img.clientWidth; cv.height = img.clientHeight;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = "#0f0"; ctx.font = "16px monospace";
  ctx.fillText(`wavers: ${s.waver_count ?? 0}`, 8, 20);
}

function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws/state`);
  ws.onopen = () => { $("conn").textContent = "live"; $("conn").className = "badge on"; };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") renderState(msg.data);
    if (msg.type === "proc") renderProc(msg.data);
  };
  ws.onclose = () => {
    $("conn").textContent = "reconnecting…"; $("conn").className = "badge off";
    setTimeout(connectWS, 1500);
  };
}

$("btn-start").onclick = async () =>
  log("start test → " + JSON.stringify(await post(`/api/test/start?mock=${$("mock").checked}`)));
$("btn-stop").onclick = async () => log("stop test → " + JSON.stringify(await post("/api/test/stop")));
$("btn-prereqs").onclick = async () =>
  log("start all → " + JSON.stringify(await post("/api/proc/group/prereqs/start")));

connectWS();
