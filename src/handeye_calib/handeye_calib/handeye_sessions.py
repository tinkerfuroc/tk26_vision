"""On-disk persistence + history for wrist hand-eye CAPTURE sessions.

A capture session is everything collected between "start fresh" and the next
reset: the per-pose samples (FK + wrist PnP + corner observations + optional FFS
depth), the per-sample capture thumbnails, and — once solved — the solve result.
Persisting it means a server restart (or a crash) no longer loses a capture, and
the operator can browse, re-open, and re-solve PAST captures the same way the
pan-tilt calibration tool browses its historical sessions.

Layout (canonical)::

    <root>/wrist_handeye_sessions/<name>/
        session.json     # meta + samples[] + (optional) "result" block
        thumbs/<i>.jpg   # per-sample capture thumbnail, i == sample index

``<root>`` defaults to ``$HANDEYE_DUMP_DIR`` or ``calibration_data`` (the same
root the flat solve-replay dumps use). These are PURE filesystem helpers — no
ROS / node / numpy deps — so they unit-test directly and the node layer just
feeds them already-serialized dicts + JPEG bytes.
"""
import json
import os
import shutil
import time

SESSIONS_SUBDIR = "wrist_handeye_sessions"


def _root(base=None):
    base = base or os.environ.get("HANDEYE_DUMP_DIR", "calibration_data")
    return os.path.join(base, SESSIONS_SUBDIR)


def _safe_name(name):
    """Reject path-traversal / hidden names so a session name can never escape
    the sessions root (the name reaches us from an HTTP path segment)."""
    name = str(name)
    if (not name or "/" in name or "\\" in name
            or name in (".", "..") or name.startswith(".")):
        raise ValueError(f"invalid session name: {name!r}")
    return name


def new_session_name(now_struct=None):
    """A fresh timestamped session name. ``now_struct`` (a ``time.struct_time``)
    is injectable so tests are deterministic."""
    return "wrist_handeye_" + time.strftime("%Y%m%d_%H%M%S",
                                            now_struct or time.localtime())


def session_dir(name, base=None):
    return os.path.join(_root(base), _safe_name(name))


def write_session(name, payload, base=None):
    """Write/overwrite ``session.json`` for ``name``; create the dir + thumbs/
    subdir if needed. Returns the session directory path."""
    d = session_dir(name, base)
    os.makedirs(os.path.join(d, "thumbs"), exist_ok=True)
    tmp = os.path.join(d, "session.json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, os.path.join(d, "session.json"))  # atomic swap
    return d


def write_thumb(name, idx, jpg_bytes, base=None):
    """Persist one sample thumbnail (JPEG bytes) at ``thumbs/<idx>.jpg``."""
    d = os.path.join(session_dir(name, base), "thumbs")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{int(idx)}.jpg"), "wb") as f:
        f.write(jpg_bytes)


def rewrite_thumbs(name, thumbs_by_idx, base=None):
    """Replace the whole thumbs/ dir from ``{idx: jpg_bytes}`` — used after a
    delete recompacts the sample indices so the on-disk files match 0..N-1."""
    d = os.path.join(session_dir(name, base), "thumbs")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    for idx, jpg in (thumbs_by_idx or {}).items():
        if jpg is not None:
            with open(os.path.join(d, f"{int(idx)}.jpg"), "wb") as f:
                f.write(jpg)


def thumb_path(name, idx, base=None):
    """Filesystem path of a stored thumbnail (may not exist)."""
    return os.path.join(session_dir(name, base), "thumbs", f"{int(idx)}.jpg")


def write_placement_thumb(name, placement_id, idx, jpg_bytes, base=None):
    """Persist one sample thumbnail at ``thumbs/<placement_id>/<idx>.jpg``."""
    d = os.path.join(session_dir(name, base), "thumbs", str(placement_id))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{int(idx)}.jpg"), "wb") as f:
        f.write(jpg_bytes)


def rewrite_placement_thumbs(name, placement_id, thumbs_by_idx, base=None):
    """Replace ``thumbs/<placement_id>/`` from ``{idx: jpg_bytes}``."""
    d = os.path.join(session_dir(name, base), "thumbs", str(placement_id))
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    for idx, jpg in (thumbs_by_idx or {}).items():
        if jpg is not None:
            with open(os.path.join(d, f"{int(idx)}.jpg"), "wb") as f:
                f.write(jpg)


def placement_thumb_path(name, placement_id, idx, base=None):
    """Filesystem path of a placement thumbnail (may not exist)."""
    return os.path.join(session_dir(name, base), "thumbs", str(placement_id), f"{int(idx)}.jpg")


def read_session(name, base=None):
    """Parse and return a session's ``session.json`` dict (raises on missing)."""
    with open(os.path.join(session_dir(name, base), "session.json")) as f:
        return json.load(f)


def _summary(name, data, mtime):
    res = data.get("result") or {}
    placements = data.get("placements")
    if placements is not None:
        n_samples_total = sum(len(p.get("samples") or []) for p in placements)
        placement_summaries = [
            {"id": p.get("id"), "label": p.get("label"),
             "n_samples": len(p.get("samples") or []),
             "has_solve": bool(p.get("result")),
             "status": (p.get("result") or {}).get("status")}
            for p in placements
        ]
        combined_res = data.get("combined_result") or {}
        n_placements = len(placements)
    else:
        n_samples = len(data.get("samples") or [])
        n_samples_total = n_samples
        placement_summaries = [
            {"id": "default", "label": None,
             "n_samples": n_samples,
             "has_solve": bool(res),
             "status": res.get("status")}
        ]
        combined_res = {}
        n_placements = 1
    return {
        "name": name,
        "mtime": mtime,
        "timestamp": data.get("timestamp"),
        "n_samples": len(data.get("samples") or []),
        "has_solve": bool(res),
        "status": res.get("status"),
        "robot": data.get("robot"),
        "calib_frame": data.get("calib_frame"),
        "n_rejected": len(res.get("rejected_sample_indices") or []),
        "n_placements": n_placements,
        "n_samples_total": n_samples_total,
        "placements": placement_summaries,
        "has_combined_solve": bool(combined_res),
        "combined_status": combined_res.get("status"),
    }


def list_sessions(base=None):
    """Newest-first list of session summaries (one per dir holding a readable
    ``session.json``). Unreadable / partial dirs are skipped, never raised."""
    root = _root(base)
    out = []
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        sj = os.path.join(root, name, "session.json")
        if not os.path.isfile(sj):
            continue
        try:
            with open(sj) as f:
                data = json.load(f)
            out.append(_summary(name, data, os.path.getmtime(sj)))
        except Exception:
            continue  # partial/corrupt dir — list the rest
    out.sort(key=lambda s: (s.get("mtime") or 0), reverse=True)
    return out


def delete_session(name, base=None):
    """Remove a session directory and everything in it. Returns True if it
    existed."""
    d = session_dir(name, base)
    if os.path.isdir(d):
        shutil.rmtree(d)
        return True
    return False
