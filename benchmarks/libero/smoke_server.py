# benchmarks/libero/smoke_server.py
"""Serve a randomly initialized pi05 with LIBERO feature shapes (no
checkpoint needed) to smoke-test the eval pipeline plumbing."""
import torch

from benchmarks.libero.libero_config import ACTION_DIM, IMAGE_KEYS, SERVER_PORT, STATE_DIM
from benchmarks.libero.serve_policy import make_app
from lerobot.configs.types import FeatureType, PolicyFeature
from vlash.policies.pi05.configuration_pi05 import PI05Config
from vlash.policies.pi05.modeling_pi05 import PI05Policy

cfg = PI05Config(
    input_features={
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
        for key in IMAGE_KEYS.values()
    }
    | {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))},
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    state_cond=True,
    device="cuda",
)
policy = PI05Policy(cfg).to("cuda").eval()
make_app(policy).run(host="127.0.0.1", port=SERVER_PORT, threaded=False)
