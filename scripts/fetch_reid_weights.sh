#!/usr/bin/env bash
# Fetch the validated MSMT17-trained osnet_ain_x1_0 ReID checkpoint into the
# torch cache, where vision_track's reid_backbone auto-discovers and loads it
# (no reid_weights_path needed). Improves lookalike discrimination over the
# imagenet-init default (separation 0.47 -> 0.57 on real Tinker crops).
#
# Idempotent: skips the download if the file is already present.
set -euo pipefail

WEIGHTS="osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
URL="https://huggingface.co/kaiyangzhou/osnet/resolve/main/${WEIGHTS}"
CACHE_DIR="${HOME}/.cache/torch/checkpoints"
DEST="${CACHE_DIR}/${WEIGHTS}"

mkdir -p "${CACHE_DIR}"

if [[ -f "${DEST}" ]]; then
    echo "ReID weights already present: ${DEST}"
    exit 0
fi

echo "Fetching MSMT17 osnet_ain_x1_0 ReID weights -> ${DEST}"
curl -fSL "${URL}" -o "${DEST}"
echo "Done: ${DEST}"
