"""End-to-end npz round-trip through the Flask test client with a
randomly initialized pi05 shaped like the LIBERO datasets (no download)."""
import io

import numpy as np
import pytest
import torch

from benchmarks.libero.libero_config import ACTION_DIM, IMAGE_KEYS, K, STATE_DIM
from benchmarks.libero.serve_policy import make_app


@pytest.fixture(scope="module")
def app():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from vlash.policies.pi05.configuration_pi05 import PI05Config
    from vlash.policies.pi05.modeling_pi05 import PI05Policy

    cfg = PI05Config(
        input_features={
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
            for key in IMAGE_KEYS.values()
        }
        | {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))},
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))
        },
        state_cond=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    policy = PI05Policy(cfg).to(cfg.device).eval()
    return make_app(policy)


def test_health(app):
    client = app.test_client()
    assert client.get("/health").get_json() == {"status": "ok"}


def test_predict_roundtrip(app):
    client = app.test_client()
    buf = io.BytesIO()
    np.savez(
        buf,
        image=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist_image=np.zeros((256, 256, 3), dtype=np.uint8),
        state=np.zeros(STATE_DIM, dtype=np.float32),
        task=np.array("put the bowl on the plate"),
    )
    resp = client.post("/predict", data=buf.getvalue())
    assert resp.status_code == 200
    out = np.load(io.BytesIO(resp.data))
    assert out["actions"].shape == (K, ACTION_DIM)
    assert out["actions"].dtype == np.float32
