# Vendored FoundationStereo + Fast-FoundationStereo

Source-only mirror of NVIDIA's FoundationStereo (CVPR 2025) and
Fast-FoundationStereo. Stripped to just what the ROS node imports:

- `core/`, `Utils.py`, `scripts/`
- LICENSE files, readme.md, model_card.md, requirements.txt

**Excluded (kept out of the workspace):**
- `pretrained_models/` (FoundationStereo, ~3 GB)
- `weights/`, `output*` (Fast-FoundationStereo TRT engines)
- `.git/`, `.venv/`, `__pycache__/`, `captures/`, `demo_data/`

Weights / TRT engines live at the `weights_root` ROS param of the
`foundation_stereo` node (default: the original reference directory at
`/home/tinker/projects/vision_tests/dualrRGB-foundationStereo`).

To refresh this vendor copy from a newer upstream, re-run the `rsync` lines
in `docs/superpowers/plans/2026-05-24-foundation-stereo.md` Task 1.

Upstream commits at vendor time: see git log for the directory.
