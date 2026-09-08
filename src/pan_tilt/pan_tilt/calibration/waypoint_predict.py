"""End-point pose predictors for waypoint pruning.

Three predictor sources, composed via ``chain_predictors`` in priority order:

* :func:`replay_predictor` — load ``T_base_ee`` (Phase 1) or
  ``T_base_camera`` (Phase 2) from a prior collection's sample file. Strictly
  the most accurate source — includes real servo offsets — and free.
* :func:`pantilt_grid_predictor` — analytical
  ``pan_tilt_model.forward_kinematics(pan, tilt, params)`` for Phase-2 cell
  similarity. Default :class:`PanTiltParams` suffices because cell similarity
  is a *relative* judgement among predicted poses.
* The plan also lists a yourdfpy-backed xArm FK predictor for first-run
  Phase-1 prediction without prior data. That dependency isn't installed in
  the venv today; the predictor is omitted on purpose. Phase-1 pruning
  requires a prior run available via :func:`replay_predictor`.

All predictors return :class:`waypoint_prune.Predicted`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .pan_tilt_model import PanTiltParams, forward_kinematics
from .utils import pose_to_matrix
from .waypoint_prune import Predicted, PredictPoseFn


# ---- replay (prior run) predictor ------------------------------------------

def replay_predictor(
    sample_path: Path | str,
    *,
    pose_key: str = "t_base_ee",
    label_key: str = "label",
) -> PredictPoseFn:
    """Build a predictor that returns the pose recorded at the same label in
    a prior calibration sample file.

    The sample file shape is the JSON written by ``calibrate_collect`` —
    either ``{"samples": [...]}`` or a bare list of sample dicts. Each entry
    must carry a ``label`` plus a ``pose_key`` block of
    ``{"translation": [x,y,z], "rotation": [qx,qy,qz,qw]}``.

    The predictor matches the *payload's* ``label`` field against the saved
    labels. A miss returns ``Predicted(None, "label not in prior run: <…>")``.

    Parameters
    ----------
    sample_path
        Path to ``phase1_handeye.json`` etc.
    pose_key
        Sample-dict key whose pose is returned. Use ``"t_base_ee"`` for
        Phase-1 hand-eye (the default) or ``"t_cam_marker_body"`` to compare
        camera-frame marker poses.
    label_key
        Key in *both* the saved samples and the payloads being pruned. Default
        ``"label"`` matches the calibrate_collect convention.
    """
    p = Path(sample_path).expanduser().resolve()
    raw = json.loads(p.read_text())
    samples = raw.get("samples") if isinstance(raw, dict) and "samples" in raw else raw
    if not isinstance(samples, list):
        raise ValueError(
            f"{p}: expected list of samples or {{'samples': [...]}}, got {type(raw).__name__}"
        )

    by_label: dict[str, np.ndarray] = {}
    for s in samples:
        label = s.get(label_key)
        pose_block = s.get(pose_key)
        if not isinstance(label, str) or not isinstance(pose_block, dict):
            continue
        try:
            T = pose_to_matrix(pose_block["translation"], pose_block["rotation"])
        except (KeyError, ValueError):
            continue
        by_label[label] = T

    if not by_label:
        raise ValueError(
            f"{p}: no usable {pose_key} entries found in {len(samples)} samples"
        )

    src = f"replay({p.name}:{pose_key})"

    def predict(_index: int, payload: dict) -> Predicted:
        label = payload.get(label_key)
        if not isinstance(label, str):
            return Predicted(None, "payload missing label")
        T = by_label.get(label)
        if T is None:
            return Predicted(None, f"label not in prior run: {label}")
        return Predicted(pose=T, source=src)

    predict.label_index = by_label  # exposed for diagnostics
    return predict


# ---- pan-tilt cell predictor (Phase 2) -------------------------------------

def pantilt_grid_predictor(
    params: Optional[PanTiltParams] = None,
) -> PredictPoseFn:
    """Predict the camera pose ``T_base_head_camera`` for a Phase-2 cell.

    Each payload must carry ``pan_deg`` and ``tilt_deg`` (the firmware-
    commanded angles). Default :class:`PanTiltParams` is used when ``params``
    is omitted; cell similarity is a relative judgement, so calibrated
    parameter values are not required.
    """
    p = params or PanTiltParams()
    src = "pantilt_grid_fk(default_params)" if params is None else "pantilt_grid_fk(operator_params)"

    def predict(_index: int, payload: dict) -> Predicted:
        try:
            pan_deg = float(payload["pan_deg"])
            tilt_deg = float(payload["tilt_deg"])
        except (KeyError, TypeError, ValueError):
            return Predicted(None, "payload missing pan_deg/tilt_deg")
        T = forward_kinematics(np.deg2rad(pan_deg), np.deg2rad(tilt_deg), p)
        return Predicted(pose=np.asarray(T, dtype=float), source=src)

    return predict


# ---- chain combinator ------------------------------------------------------

def chain_predictors(predictors: list[PredictPoseFn]) -> PredictPoseFn:
    """Try each predictor in order; return the first non-failure.

    The aggregated source string lists which predictor fired first; on full
    failure the source carries the *last* predictor's reason (so the operator
    sees the most-specific message in the UI).
    """
    if not predictors:
        return _no_predictor

    def predict(index: int, payload: dict) -> Predicted:
        last_reason = "no predictors configured"
        for pred in predictors:
            try:
                out = pred(index, payload)
            except Exception as exc:  # pragma: no cover - defensive
                last_reason = f"{getattr(pred, '__name__', repr(pred))} raised: {exc!r}"
                continue
            if not isinstance(out, Predicted):
                last_reason = f"predictor returned {type(out).__name__}, not Predicted"
                continue
            if out.ok:
                return out
            last_reason = out.source
        return Predicted(None, last_reason)

    return predict


def _no_predictor(_index: int, _payload: dict) -> Predicted:
    return Predicted(None, "no predictors configured")
