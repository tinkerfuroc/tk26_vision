"""Pure, ROS-free helpers + inline UI for handeye_web. No rclpy/fastapi here."""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def tf_to_matrix(translation_xyz, quaternion_xyzw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(quaternion_xyzw, float)).as_matrix()
    T[:3, 3] = np.asarray(translation_xyz, float)
    return T


def matrix_to_xyz_rpy(T):
    T = np.asarray(T, float)
    xyz = T[:3, 3].tolist()
    rpy = R.from_matrix(T[:3, :3]).as_euler('xyz').tolist()  # URDF fixed-axis convention
    return xyz, rpy


def charuco_to_sample_arrays(charuco_corners, charuco_ids):
    px = np.asarray(charuco_corners, float).reshape(-1, 2)
    idx = np.asarray(charuco_ids).reshape(-1).astype(int)
    return px, idx


_GATE_COLORS = {"PASS": "#1a9850", "WARN": "#f59e0b", "FAIL": "#d73027"}


def gate_color(status):
    return _GATE_COLORS.get(status, "#888888")


def state_payload(camera_connected, intrinsics_ok, num_samples, last_detection, status_msg):
    return {
        "camera_connected": bool(camera_connected),
        "intrinsics_ok": bool(intrinsics_ok),
        "num_samples": int(num_samples),
        "last_detection": last_detection,
        "status_msg": status_msg,
    }


def solve_payload(res):
    xyz, rpy = matrix_to_xyz_rpy(res.X)
    return {
        "status": res.status,
        "X_xyz": xyz,
        "X_rpy": rpy,
        "heldout_metrics": res.heldout_metrics,
        "train_metrics": res.train_metrics,
    }


def encode_jpeg(bgr):
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def placeholder_jpeg(text="no camera", size=(480, 640)):
    img = np.full((size[0], size[1], 3), 40, np.uint8)
    cv2.putText(img, text, (20, size[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (200, 200, 200), 2, cv2.LINE_AA)
    return encode_jpeg(img)


def draw_charuco_overlay(bgr, corners_xy):
    out = bgr.copy()
    for (x, y) in np.asarray(corners_xy, float).reshape(-1, 2):
        cv2.circle(out, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1, cv2.LINE_AA)
    return out


INDEX_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>handeye_web</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee;display:flex;gap:16px;padding:16px}
#left{flex:0 0 660px}#right{flex:1}img{width:640px;background:#000;border:1px solid #333}
button{background:#2563eb;color:#fff;border:0;padding:8px 12px;border-radius:6px;margin:4px 0;cursor:pointer}
textarea{width:100%;height:60px;background:#1b1b1b;color:#eee;border:1px solid #333}
pre{background:#1b1b1b;padding:8px;border-radius:6px;white-space:pre-wrap;max-height:40vh;overflow:auto}
#banner{font-size:20px;font-weight:700;padding:8px;border-radius:6px;text-align:center}
.row{margin:8px 0}</style></head><body>
<div id="left"><img id="cam" src="/api/frame.jpg"><div class="row" id="status">…</div>
<div class="row"><textarea id="joints" placeholder="7 joint values, comma-separated"></textarea>
<button onclick="move()">Move arm</button></div>
<div class="row"><button onclick="post('/api/capture')">Capture pose</button>
<button onclick="post('/api/solve')">Solve</button>
<button onclick="post('/api/promote')">Promote</button></div>
<div id="banner"></div></div>
<div id="right"><h3>Result</h3><pre id="out">—</pre></div>
<script>
const out=document.getElementById('out'),banner=document.getElementById('banner');
function show(o){out.textContent=JSON.stringify(o,null,2);
  if(o.status){banner.textContent=o.status;
    banner.style.background=o.status==='PASS'?'#1a9850':o.status==='WARN'?'#f59e0b':o.status==='FAIL'?'#d73027':'#444';}}
async function post(u){try{const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});show(await r.json());}catch(e){show({error:String(e)});}}
async function move(){const j=document.getElementById('joints').value.split(',').map(Number);
  const r=await fetch('/api/move',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({joints:j})});show(await r.json());}
async function poll(){try{const r=await fetch('/api/state');const s=await r.json();
  document.getElementById('status').textContent=`camera:${s.camera_connected} K:${s.intrinsics_ok} samples:${s.num_samples} — ${s.status_msg}`;}catch(e){}}
setInterval(()=>{document.getElementById('cam').src='/api/frame.jpg?t='+Date.now();},200);
setInterval(poll,1000);poll();
</script></body></html>"""
