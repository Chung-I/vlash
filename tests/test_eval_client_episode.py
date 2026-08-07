"""run_episode's heartbeat cadence (no LIBERO/network needed -- fakes both
the env and the remote policy).

This is the mechanism that lets the subprocess-isolation parent tell a
slow-but-alive libero_10 episode (~30+ min at observed server latency)
apart from an actually-dead/stuck worker without waiting for the whole
episode: see benchmarks/libero/eval_client.py's EPISODE_TIMEOUT_S comment.
"""
import numpy as np

import benchmarks.libero.eval_client as eval_client_module
from benchmarks.libero.eval_client import HEARTBEAT_EVERY_N_STEPS, run_episode


class _FakeRemotePolicy:
    """Stands in for RemotePolicy: no HTTP, returns a constant k=5 chunk."""

    def __init__(self, port, task):
        pass

    def __call__(self, images, state, task):
        return np.zeros((5, 7), dtype=np.float32)


class _FakeEnv:
    def __init__(self, done_at_step=None):
        self.done_at_step = done_at_step
        self.n_steps = 0

    def reset(self):
        pass

    def set_init_state(self, init_state):
        return self._obs()

    def step(self, action):
        self.n_steps += 1
        done = self.done_at_step is not None and self.n_steps >= self.done_at_step
        return self._obs(), 0.0, done, {}

    def _obs(self):
        return {
            "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }


def test_run_episode_calls_heartbeat_every_n_steps(monkeypatch):
    monkeypatch.setattr(eval_client_module, "RemotePolicy", _FakeRemotePolicy)
    monkeypatch.setattr(eval_client_module, "SETTLE_STEPS", 0)
    env = _FakeEnv(done_at_step=None)  # never "done" -> runs the full max_steps
    calls = []
    ok = run_episode(
        env,
        init_state=None,
        task_str="task",
        delay=0,
        max_steps=45,
        port=1234,
        heartbeat=lambda step: calls.append(step),
    )
    assert ok is False
    assert calls == [s for s in range(45) if s % HEARTBEAT_EVERY_N_STEPS == 0]


def test_run_episode_returns_true_when_env_reports_done(monkeypatch):
    monkeypatch.setattr(eval_client_module, "RemotePolicy", _FakeRemotePolicy)
    monkeypatch.setattr(eval_client_module, "SETTLE_STEPS", 0)
    env = _FakeEnv(done_at_step=3)
    ok = run_episode(
        env, init_state=None, task_str="task", delay=0, max_steps=50, port=1234
    )
    assert ok is True


def test_run_episode_without_heartbeat_callback_is_a_noop(monkeypatch):
    monkeypatch.setattr(eval_client_module, "RemotePolicy", _FakeRemotePolicy)
    monkeypatch.setattr(eval_client_module, "SETTLE_STEPS", 0)
    env = _FakeEnv(done_at_step=None)
    # heartbeat defaults to None -- must not raise
    ok = run_episode(env, init_state=None, task_str="task", delay=0, max_steps=10, port=1234)
    assert ok is False
