"""Torch RTC port (arXiv 2506.07339): the same two invariants that pinned the
openpi/JAX port (openpi tests/test_rtc.py), for the cross-check implementation.

The full PI05 backbone is a 3B PaliGemma -- untestable on CPU -- so these tests
drive the REAL sample_actions / sample_actions_rtc loop code (unbound methods of
PI05Model) through a stub whose _denoise_step is a deterministic linear map.
That exercises everything the port added: the integration loop, the one-step
denoiser under the tau: 1->0 convention, the vjp correction, the guidance
constants, and -- critically -- the correction SIGN (dt<0), which a wrong-sign
port would fail exactly as it did once in the JAX port.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vlash.policies.pi05.modeling_pi05 import PI05Model
from vlash.policies.pi05.utils import rtc_prefix_weights

H, AD = 8, 4


class StubModel:
    """Minimal host for PI05Model's sampling methods with a fake denoiser."""

    def __init__(self):
        self.config = SimpleNamespace(chunk_size=H, max_action_dim=AD, num_inference_steps=6)
        g = torch.Generator().manual_seed(0)
        self._a = 0.3 * torch.randn(AD, AD, generator=g)

    def sample_noise(self, shape, device):
        g = torch.Generator().manual_seed(42)
        return torch.normal(0.0, 1.0, size=shape, dtype=torch.float32, generator=g).to(device)

    def _prefill_prefix(self, images, img_masks, tokens, masks):
        return None, None

    # deterministic, x-dependent, differentiable "velocity field"
    def _denoise_step(self, prefix_pad_masks, prefix_att_masks, state, x_t, timestep):
        t = timestep[0].to(torch.float32)
        return x_t @ self._a + 0.1 * t

    def denoise_step(self, *args):
        with torch.no_grad():
            return self._denoise_step(*args)


def _stub_inputs():
    tokens = torch.zeros(1, 3, dtype=torch.long)
    return (None, None, tokens, None, None)  # images, img_masks, tokens, masks, state


def test_prefix_weights_match_reference_docstring():
    w = rtc_prefix_weights(2, 6, 10, "linear")
    np.testing.assert_allclose(w, [1, 1, 4 / 5, 3 / 5, 2 / 5, 1 / 5, 0, 0, 0, 0], atol=1e-9)
    e = rtc_prefix_weights(3, 8, 10, "exp")
    assert np.all(e[:3] == 1.0) and np.all(e[8:] == 0.0)
    assert np.all(np.diff(e[2:9]) <= 1e-12)
    assert np.all(rtc_prefix_weights(5, 0, 10, "exp") == 0.0)


def test_zero_weights_reduce_to_plain_sampler():
    """With all-zero weights the correction is exactly zero, so the guided loop
    must reproduce sample_actions bit-for-bit given the same noise."""
    stub = StubModel()
    noise = stub.sample_noise((1, H, AD), "cpu")
    plain = PI05Model.sample_actions(stub, *_stub_inputs(), noise=noise.clone())
    rtc = PI05Model.sample_actions_rtc(
        stub,
        *_stub_inputs(),
        prev_actions=torch.zeros(1, H, AD),
        prefix_weights=torch.zeros(H),
        noise=noise.clone(),
    )
    np.testing.assert_allclose(rtc.numpy(), plain.numpy(), atol=1e-6)


def test_guidance_pulls_toward_prev_chunk():
    """Strong weights on early positions must land the guided sample closer to
    the previous chunk there than the unguided sample. A sign error in the
    correction makes this fail in the opposite direction."""
    stub = StubModel()
    noise = stub.sample_noise((1, H, AD), "cpu")
    plain = PI05Model.sample_actions(stub, *_stub_inputs(), noise=noise.clone())
    prev = plain + 0.5
    w = torch.from_numpy(rtc_prefix_weights(4, 8, H, "exp")).to(torch.float32)
    rtc = PI05Model.sample_actions_rtc(
        stub, *_stub_inputs(), prev_actions=prev, prefix_weights=w, noise=noise.clone()
    )
    d_guided = (rtc[:, :4] - prev[:, :4]).abs().mean().item()
    d_plain = (plain[:, :4] - prev[:, :4]).abs().mean().item()
    assert np.isfinite(d_guided)
    assert d_guided < d_plain, f"guidance did not pull toward prev: {d_guided=} {d_plain=}"


def test_overlap_executor_protocol():
    """OverlapChunkExecutor matches the openpi client's slicing exactly."""
    import sys, pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "benchmarks" / "libero"))
    from executor import OverlapChunkExecutor

    calls = []

    def fake_predict(images, state, task, rtc=None, full=False):
        r = len(calls)
        calls.append(rtc)
        return np.arange(10, dtype=np.float64)[:, None] + 100.0 * r

    ex = OverlapChunkExecutor(fake_predict, k=5, delay=2, arm="rtc", env_id=7)
    acts = [float(ex.act({}, None, "t")[0]) for _ in range(15)]
    assert acts[:5] == [0, 1, 2, 3, 4]
    assert acts[5:10] == [5, 6, 102, 103, 104]
    assert acts[10:15] == [105, 106, 202, 203, 204]
    assert calls[0] == {"env_id": 7, "delay": 2, "executed": 5}
    with pytest.raises(AssertionError):
        OverlapChunkExecutor(fake_predict, k=5, delay=6, arm="naive", env_id=1)
