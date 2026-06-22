"""Vision-log session-directory resolver — shared with the rest of tk26.

Matches the resolution order documented in src/tk26_vision/CLAUDE.md:
  1. $TINKER_VISION_SESSION_TS (must match YYYYmmdd_HHMMSS).
  2. Newest existing <base>/<YYYYmmdd_HHMMSS>/ subdir by mtime — lets
     late-spawned standalone nodes join the active session.
  3. Fresh strftime cold-start.
"""

from __future__ import annotations

import os
import re
import time

_TS_RE = re.compile(r"^\d{8}_\d{6}$")


def resolve_session_dir(base: str) -> str:
    """Return the active session directory, creating it if necessary."""
    os.makedirs(base, exist_ok=True)

    env_ts = os.environ.get("TINKER_VISION_SESSION_TS", "")
    if _TS_RE.match(env_ts):
        out = os.path.join(base, env_ts)
        os.makedirs(out, exist_ok=True)
        return out

    candidates = []
    for entry in os.listdir(base):
        path = os.path.join(base, entry)
        if _TS_RE.match(entry) and os.path.isdir(path):
            candidates.append((os.path.getmtime(path), path))
    if candidates:
        candidates.sort()
        return candidates[-1][1]

    fresh = os.path.join(base, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(fresh, exist_ok=True)
    return fresh
