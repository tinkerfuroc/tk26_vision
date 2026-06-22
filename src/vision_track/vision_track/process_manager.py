# Copyright 2026 Tinker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixed-allowlist subprocess supervisor for the track_web dashboard.

The track_web webui starts/stops the follow-person demo components (audio,
follow_server, and the follow-person BT — vision-only or with-nav) on demand.
This module is the standalone, ROS-free, unit-testable supervisor that does the
spawning.

SECURITY: the public API takes a *name* validated against a fixed module-level
REGISTRY, never a command. The browser can therefore only ever launch one of a
known, vetted set of argv lists — there is no path by which a request body
becomes part of a spawned command line.

Each child is started in its OWN process group (``start_new_session=True``) so
that the entire ``ros2 launch`` process tree can be signalled with a single
``killpg`` — SIGTERM-then-SIGKILL with a grace window.

Thread-safety: uvicorn serves HTTP handlers on worker threads, so several may
call start/stop/status concurrently. Every public method holds a single
``threading.Lock`` for the whole operation.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time

# Fixed allowlist. Keys are the only values the API will accept; the argv lists
# are never built from caller input.
REGISTRY = {
    "audio":         ["ros2", "launch", "audio_pakage", "audio.launch.py"],
    "follow_server": ["ros2", "run", "following", "follow_server",
                      "--ros-args", "-p", "working_frame:=map"],
    "bt_vision":     ["ros2", "run", "behavior_tree", "follow-person", "--no-nav"],
    "bt_nav":        ["ros2", "run", "behavior_tree", "follow-person"],
}

# Fixed group allowlist for the two follow demo modes. Members are REGISTRY keys
# started in listed order (audio + TTS up, then follow_server, then the BT).
GROUPS = {
    "follow_vision": ["audio", "bt_vision"],
    "follow_nav":    ["audio", "follow_server", "bt_nav"],
}


class ProcessManager:
    """Supervise a fixed set of named subprocesses (start/stop/status)."""

    def __init__(self, registry=REGISTRY, groups=GROUPS, stagger_sec=1.5):
        """Store the allowlist and init per-name process / returncode caches.

        Args:
            registry: mapping of name -> argv list. Defaults to the module
                REGISTRY. A copy is stored so later edits to the passed dict
                don't mutate this manager's allowlist.
            groups: mapping of group name -> list of REGISTRY keys started as a
                unit. Defaults to the module GROUPS. A copy is stored so later
                edits to the passed dict don't mutate this manager's groups.
            stagger_sec: seconds to sleep BETWEEN successive members when
                starting a group, so a downstream node has a moment to come up
                before the next one launches. Defaults to 1.5.
        """
        self._registry = dict(registry)
        self._groups = dict(groups)
        self._stagger_s = float(stagger_sec)
        self._procs: dict[str, subprocess.Popen] = {}
        self._last_rc: dict[str, int | None] = {}
        self._lock = threading.Lock()
        self.term_timeout_s = 5.0

    # -- public API --------------------------------------------------------

    def start(self, name) -> dict:
        """Start ``name`` if registered and not already running.

        Idempotent: a second start while the process is alive returns the
        current status without spawning a duplicate. Never raises on an
        unknown name (returns an error dict) or a spawn failure (returns an
        error dict).
        """
        with self._lock:
            if name not in self._registry:
                return self._unknown(name)
            proc = self._procs.get(name)
            if proc is not None and proc.poll() is None:
                # Already running — do not spawn a second one.
                return self._status_locked(name)
            try:
                # start_new_session=True -> child leads its own process group,
                # so killpg(getpgid(pid)) reaches the whole ros2 launch tree.
                # stdout/stderr inherit the parent's (None) to avoid a pipe
                # that nobody drains (which would deadlock the child).
                proc = subprocess.Popen(
                    self._registry[name],
                    start_new_session=True,
                    env=os.environ.copy(),
                    stdout=None,
                    stderr=None,
                )
            except Exception as exc:  # spawn failure: report, never crash
                return {"name": name, "error": f"failed to start: {exc}"}
            self._procs[name] = proc
            self._last_rc[name] = None
            return self._status_locked(name)

    def stop(self, name) -> dict:
        """Stop ``name`` with SIGTERM, escalating to SIGKILL after a grace.

        Never raises on an unknown name (error dict) or an already-stopped
        process (returns status).
        """
        with self._lock:
            if name not in self._registry:
                return self._unknown(name)
            proc = self._procs.get(name)
            if proc is None or proc.poll() is not None:
                # Not running (or already exited) — reap + report.
                return self._status_locked(name)
            self._terminate_locked(proc)
            self._last_rc[name] = proc.poll()
            return self._status_locked(name)

    def status(self, name) -> dict:
        """Return the current status dict for ``name`` (reaps if exited)."""
        with self._lock:
            if name not in self._registry:
                return self._unknown(name)
            return self._status_locked(name)

    def status_all(self) -> dict:
        """Return ``{name: status(name)}`` for every registered name."""
        with self._lock:
            return {name: self._status_locked(name)
                    for name in self._registry}

    def shutdown_all(self) -> None:
        """Stop every currently-running process. Must not raise.

        Used when the dashboard exits; best-effort, errors are swallowed so a
        single stuck child can't block teardown of the rest.
        """
        with self._lock:
            for name in list(self._registry):
                proc = self._procs.get(name)
                if proc is None or proc.poll() is not None:
                    continue
                try:
                    self._terminate_locked(proc)
                    self._last_rc[name] = proc.poll()
                except Exception:
                    pass

    # -- group API ---------------------------------------------------------

    def start_group(self, group) -> list | dict:
        """Start every member of ``group`` in listed order, staggered.

        Returns the list of per-member status dicts, or an error dict for an
        unknown group. Never raises. Each member goes through ``start`` (its own
        lock acquisition), so the stagger sleep happens BETWEEN members, outside
        the lock — the dashboard stays responsive.
        """
        if group not in self._groups:
            return {"group": group, "error": f"unknown group '{group}'"}
        out = []
        members = self._groups[group]
        for i, name in enumerate(members):
            out.append(self.start(name))
            if self._stagger_s and i < len(members) - 1:
                time.sleep(self._stagger_s)
        return out

    def stop_group(self, group) -> list | dict:
        """Stop every member of ``group`` in REVERSE order. Never raises."""
        if group not in self._groups:
            return {"group": group, "error": f"unknown group '{group}'"}
        return [self.stop(name) for name in reversed(self._groups[group])]

    # -- internals (all called with self._lock held) -----------------------

    def _unknown(self, name) -> dict:
        return {"name": name, "error": f"unknown process '{name}'"}

    def _status_locked(self, name) -> dict:
        """Poll the Popen to reap finished children, cache + report state."""
        proc = self._procs.get(name)
        running = False
        pid = None
        if proc is not None:
            rc = proc.poll()
            if rc is None:
                running = True
                pid = proc.pid
            else:
                # Finished: cache returncode, forget the dead handle.
                self._last_rc[name] = rc
                self._procs.pop(name, None)
        return {
            "name": name,
            "running": running,
            "pid": pid,
            "returncode": self._last_rc.get(name),
        }

    def _terminate_locked(self, proc: subprocess.Popen) -> None:
        """SIGTERM the process group, wait, then SIGKILL if still alive."""
        self._signal_group(proc, signal.SIGTERM)
        deadline = time.time() + self.term_timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.05)
        if proc.poll() is None:
            self._signal_group(proc, signal.SIGKILL)
            try:
                proc.wait(timeout=self.term_timeout_s)
            except Exception:
                pass

    @staticmethod
    def _signal_group(proc: subprocess.Popen, sig) -> None:
        """Send ``sig`` to the child's process group; tolerate it being gone."""
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            # Already dead, or we no longer own it — nothing to do.
            pass
