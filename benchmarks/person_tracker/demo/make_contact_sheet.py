#!/usr/bin/env python3
"""Labeled ReID contact sheet — demonstrates the production embedding discriminates.

Loads real Tinker Orbbec person crops (from vision_log feature-extraction dumps),
embeds each with the PRODUCTION ReID backbone (OSNet-AIN x1.0, MSMT17-trained,
fp16 — the exact path person_track_server uses), and renders a contact sheet:

  - a row of the labeled crops, each annotated with its same-identity score
    (crop vs its own horizontal flip — a guaranteed same-person pair), and
  - the crop-vs-crop cosine-similarity matrix (cross-identity similarities).

The visual story: every crop matches its own flipped view ~0.95+ (same person,
new view) while different crops score far lower — i.e. the embedding tells people
apart. Contrast with the pre-overhaul random head, which produced noise here.

Run (main session, CUDA + cached MSMT17 weights):
    .venv-vision-main/bin/python benchmarks/person_tracker/demo/make_contact_sheet.py
Outputs reid_discrimination_contact_sheet.png next to this script.
"""
from __future__ import annotations

import glob
import importlib.util
import os

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WT = "/home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker"
BACKBONE_PY = f"{WT}/src/vision_track/vision_track/reid/reid_backbone.py"
VLOG = "/home/tinker/tk25_ws/vision_log"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "reid_discrimination_contact_sheet.png")

# Two crops from each of several distinct recording sessions (different setups →
# diverse identities/clothing). Session tag is the label.
SESSIONS = [
    ("17:59", f"{VLOG}/20260502_175907"),
    ("17:20", f"{VLOG}/20260502_172005"),
    ("19:14", f"{VLOG}/20260502_191402"),
    ("19:22", f"{VLOG}/20260502_192215 copy"),
]
PER_SESSION = 2


def _load_backbone():
    spec = importlib.util.spec_from_file_location("reid_backbone", BACKBONE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Production defaults: osnet_ain_x1_0, fp16 on CUDA, MSMT17 auto-discovered.
    return mod.build_reid_backbone("osnet_ain_x1_0", device="cuda", fp16=True)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    bk = _load_backbone()
    crops, labels = [], []
    for tag, d in SESSIONS:
        files = sorted(glob.glob(os.path.join(d, "*feature_extraction_crop*.jpg")))
        for i, f in enumerate(files[:PER_SESSION]):
            bgr = cv2.imread(f)
            if bgr is None:
                continue
            crops.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            labels.append(f"{tag}#{i}")
    n = len(crops)
    embs = [bk.extract_features(c) for c in crops]
    flips = [bk.extract_features(c[:, ::-1, :].copy()) for c in crops]
    self_sim = [_cos(embs[i], flips[i]) for i in range(n)]              # same id
    mat = np.array([[_cos(embs[i], embs[j]) for j in range(n)] for i in range(n)])
    cross = [mat[i, j] for i in range(n) for j in range(n) if i != j]   # diff crops

    # ---- render ----
    fig = plt.figure(figsize=(2.0 * n, 2.0 * n + 3.2))
    gs = fig.add_gridspec(2, n, height_ratios=[2.4, 2.0 * n], hspace=0.30)
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(crops[i])
        ax.set_title(f"{labels[i]}\nself-flip {self_sim[i]:.2f}", fontsize=10)
        ax.axis("off")

    axm = fig.add_subplot(gs[1, :])
    im = axm.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
    axm.set_xticks(range(n)); axm.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axm.set_yticks(range(n)); axm.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            axm.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                     color="white" if mat[i, j] < 0.6 else "black", fontsize=9)
    axm.set_title("crop-vs-crop cosine similarity (off-diagonal = different crops)",
                  fontsize=11)
    fig.colorbar(im, ax=axm, fraction=0.025, pad=0.02)

    pos_m, cross_m = float(np.mean(self_sim)), float(np.mean(cross))
    fig.suptitle(
        "Tinker person-tracker ReID embedding — OSNet-AIN x1.0 (MSMT17-trained, fp16)\n"
        f"same-identity (crop vs flip) mean={pos_m:.3f}   |   "
        f"cross-crop mean={cross_m:.3f}   |   separation={pos_m - cross_m:.3f}   "
        f"(n={n} real Orbbec crops)",
        fontsize=13, y=0.99,
    )
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"n={n}  self-flip mean={pos_m:.3f}  cross mean={cross_m:.3f}  "
          f"separation={pos_m - cross_m:.3f}")


if __name__ == "__main__":
    main()
