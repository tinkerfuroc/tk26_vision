import numpy as np
import pytest

torchreid = pytest.importorskip("torchreid")  # skip offline / if not installed

from vision_track.reid.reid_backbone import OSNetBackbone, build_reid_backbone


# osnet_ain_x1_0 is the production default and the only variant pre-cached in
# this environment (~/.cache/torch/checkpoints). Other variants would require a
# Google-Drive download, which is unreachable offline — so the integration test
# runs for real against the cached default rather than skipping.
_CACHED_BACKBONE = "osnet_ain_x1_0"


@pytest.fixture(scope="module")
def backbone():
    return build_reid_backbone(_CACHED_BACKBONE, device="cpu")


def test_extract_features_shape_and_l2_norm(backbone):
    crop = (np.random.rand(200, 80, 3) * 255).astype(np.uint8)
    feat = backbone.extract_features(crop)
    assert feat.ndim == 1
    assert feat.dtype == np.float32
    assert feat.shape[0] == backbone.feature_dim
    assert backbone.feature_dim > 0
    assert abs(float(np.linalg.norm(feat)) - 1.0) < 1e-4


def test_feature_dim_is_stable_across_calls(backbone):
    a = backbone.extract_features((np.random.rand(150, 60, 3) * 255).astype(np.uint8))
    b = backbone.extract_features((np.random.rand(300, 120, 3) * 255).astype(np.uint8))
    assert a.shape == b.shape == (backbone.feature_dim,)


def test_degenerate_crop_returns_zero_vector(backbone):
    # A 1px-tall crop is degenerate; the contract is a zero vector of the right dim.
    feat = backbone.extract_features(np.zeros((1, 1, 3), dtype=np.uint8))
    assert feat.shape == (backbone.feature_dim,)
    assert float(np.linalg.norm(feat)) == 0.0


def test_build_unknown_backbone_raises():
    with pytest.raises(ValueError):
        build_reid_backbone("not_a_real_backbone", device="cpu")


def test_osnet_backbone_type(backbone):
    assert isinstance(backbone, OSNetBackbone)


def test_default_backbone_is_osnet_ain_x1_0_512d():
    # The production default: osnet_ain_x1_0, imagenet-init (no reid_weights_path),
    # 512-dim. Weights are cached, so this builds + extracts for real.
    bb = build_reid_backbone("osnet_ain_x1_0", device="cpu")
    assert bb.feature_dim == 512
    crop = (np.random.rand(220, 90, 3) * 255).astype(np.uint8)
    feat = bb.extract_features(crop)
    assert feat.shape == (512,)
    assert abs(float(np.linalg.norm(feat)) - 1.0) < 1e-4


def test_person_reid_model_uses_backbone():
    from vision_track.reid.reid import PersonReIDModel
    m = PersonReIDModel(device="cpu", backbone_name=_CACHED_BACKBONE)
    crop = (np.random.rand(180, 70, 3) * 255).astype(np.uint8)
    feat = m.extract_features(crop)
    assert feat.shape == (m.feature_dim,)
    assert abs(float(np.linalg.norm(feat)) - 1.0) < 1e-4
