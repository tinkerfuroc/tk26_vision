"""Pluggable ReID feature backbones for PersonReIDModel.

The OSNet path loads genuine pretrained weights via torchreid's pretrained-model
cache (downloaded once to ~/.cache/torch/checkpoints, then reused). This replaces
the legacy random-head path in reid.py, whose deep term was an untrained
projection of ImageNet features (channel_attention / bottleneck / part_bottlenecks
were random nn.Modules with no checkpoint).

Weight strategy
---------------
``build_model(..., pretrained=True)`` loads only **imagenet**-initialized OSNet
weights — the PyPI torchreid 0.2.5 wheel embeds no ReID-trained (Market/MSMT)
download URLs. ImageNet-OSNet already removes the random-head defect (the real
win) but is not ReID-discriminatively trained. To upgrade, pass a
``reid_weights_path`` pointing at a Market/MSMT-trained ``osnet_ain_x1_0``
checkpoint; it is loaded via torchreid's ``load_pretrained_weights`` *after*
building, overriding the imagenet init. Empty path ⇒ keep imagenet.
"""
import logging
import os
from typing import Optional, Protocol

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# torchreid input convention for person ReID: HxW = 256x128 (h>w).
_REID_H, _REID_W = 256, 128
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Supported OSNet variants and their published embedding dims.
_OSNET_DIMS = {
    "osnet_ain_x1_0": 512,
    "osnet_x1_0": 512,
    "osnet_x0_75": 512,
    "osnet_x0_5": 512,
    "osnet_x0_25": 512,
}


class ReIDBackbone(Protocol):
    feature_dim: int
    def extract_features(self, crop: np.ndarray) -> np.ndarray: ...


def _resolve_build_model():
    """torchreid namespaces build_model differently across wheels.

    The PyPI torchreid 0.2.5 wheel exposes it at ``torchreid.reid.models``
    (and re-exports it as the attribute ``torchreid.models.build_model``, which
    is *not* an importable submodule). Newer deep-person-reid layouts expose
    ``torchreid.models``. Try both, preferring the 0.2.5 layout.
    """
    try:
        from torchreid.reid.models import build_model  # 0.2.5 wheel layout
        return build_model
    except Exception:  # pragma: no cover - version-dependent
        from torchreid.models import build_model  # newer / re-export layout
        return build_model


def _resolve_load_pretrained_weights():
    """torchreid's checkpoint loader, namespaced under torchreid.reid.utils
    in the 0.2.5 wheel (torchreid.utils in newer layouts)."""
    try:
        from torchreid.reid.utils import load_pretrained_weights  # 0.2.5 wheel
        return load_pretrained_weights
    except Exception:  # pragma: no cover - version-dependent
        from torchreid.utils import load_pretrained_weights
        return load_pretrained_weights


class OSNetBackbone:
    """OSNet feature extractor returning L2-normalized embeddings."""

    def __init__(
        self,
        backbone_name: str,
        device: str = "cpu",
        reid_weights_path: str = "",
    ):
        if backbone_name not in _OSNET_DIMS:
            raise ValueError(
                f"Unknown ReID backbone '{backbone_name}'. "
                f"Supported: {sorted(_OSNET_DIMS)}"
            )
        self.backbone_name = backbone_name
        self.device = device
        self.feature_dim = _OSNET_DIMS[backbone_name]

        build_model = _resolve_build_model()
        # num_classes is irrelevant for feature extraction; pretrained=True
        # triggers the cached imagenet-weight download (torchreid pulls OSNet
        # weights from its hosted mirror on first use, then caches them).
        model = build_model(
            name=backbone_name,
            num_classes=1,
            pretrained=True,
        )

        # Optional upgrade: load a ReID-trained (Market/MSMT) checkpoint over the
        # imagenet init. This is the recommended path for maximal lookalike
        # discrimination — config change only, no code change.
        if reid_weights_path:
            if os.path.isfile(reid_weights_path):
                load_pretrained_weights = _resolve_load_pretrained_weights()
                load_pretrained_weights(model, reid_weights_path)
                logger.info(
                    "Loaded ReID-trained weights for %s from %s (overrides imagenet init)",
                    backbone_name,
                    reid_weights_path,
                )
            else:
                logger.warning(
                    "reid_weights_path '%s' does not exist; keeping imagenet init for %s",
                    reid_weights_path,
                    backbone_name,
                )
        else:
            logger.info(
                "Built %s with imagenet-init weights (no reid_weights_path set)",
                backbone_name,
            )

        model.eval()
        model.to(self.device)
        self.model = model

    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return np.zeros(self.feature_dim, dtype=np.float32)

        resized = cv2.resize(crop, (_REID_W, _REID_H))
        norm = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor)            # [1, feature_dim]
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy().flatten().astype(np.float32)


def build_reid_backbone(
    backbone_name: str,
    device: str = "cpu",
    reid_weights_path: str = "",
) -> "ReIDBackbone":
    """Factory for the configured ReID backbone. Currently OSNet only.

    Args:
        backbone_name: OSNet variant (e.g. 'osnet_ain_x1_0', 'osnet_x0_25').
        device: torch device string.
        reid_weights_path: optional path to a ReID-trained checkpoint that
            overrides the imagenet init; empty ⇒ keep imagenet.
    """
    return OSNetBackbone(
        backbone_name, device=device, reid_weights_path=reid_weights_path
    )
