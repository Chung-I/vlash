"""Unit tests for inject_dataset_stats (benchmarks/libero/serve_policy.py).

Builds a small randomly-initialized pi05 policy the same way
tests/test_serve_policy_roundtrip.py does, but forces device="cpu" -
construction-only, and the local GPU may be occupied by an unrelated
vLLM process. No network access: stats are hand-built and passed via the
`stats=` seam instead of a real `repo_id` lookup.
"""
import numpy as np
import pytest
import torch

from benchmarks.libero.libero_config import ACTION_DIM, IMAGE_KEYS, STATE_DIM
from benchmarks.libero.serve_policy import inject_dataset_stats


@pytest.fixture(scope="module")
def policy():
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
        device="cpu",
    )
    return PI05Policy(cfg).to(cfg.device).eval()


def _fake_stats():
    return {
        "observation.state": {
            "mean": np.arange(STATE_DIM, dtype=np.float32),
            "std": np.ones(STATE_DIM, dtype=np.float32) * 2.0,
        },
        "action": {
            "mean": np.arange(ACTION_DIM, dtype=np.float32) * 0.1,
            "std": np.ones(ACTION_DIM, dtype=np.float32) * 0.5,
        },
    }


def test_buffers_start_as_inf(policy):
    assert torch.isinf(policy.normalize_inputs.buffer_observation_state["mean"]).all()
    assert torch.isinf(policy.normalize_targets.buffer_action["mean"]).all()
    assert torch.isinf(policy.unnormalize_outputs.buffer_action["mean"]).all()


def test_inject_populates_buffers(policy):
    stats = _fake_stats()
    inject_dataset_stats(policy, stats=stats)

    state_buf = policy.normalize_inputs.buffer_observation_state
    assert torch.allclose(state_buf["mean"], torch.as_tensor(stats["observation.state"]["mean"]))
    assert torch.allclose(state_buf["std"], torch.as_tensor(stats["observation.state"]["std"]))

    targets_buf = policy.normalize_targets.buffer_action
    assert torch.allclose(targets_buf["mean"], torch.as_tensor(stats["action"]["mean"]))
    assert torch.allclose(targets_buf["std"], torch.as_tensor(stats["action"]["std"]))

    unnorm_buf = policy.unnormalize_outputs.buffer_action
    assert torch.allclose(unnorm_buf["mean"], torch.as_tensor(stats["action"]["mean"]))
    assert torch.allclose(unnorm_buf["std"], torch.as_tensor(stats["action"]["std"]))


def test_second_injection_raises(policy):
    with pytest.raises(RuntimeError, match="already populated"):
        inject_dataset_stats(policy, stats=_fake_stats())
