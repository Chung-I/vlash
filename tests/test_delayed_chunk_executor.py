"""Timing semantics of emulated async delay.

With k=5, delay=d: the chunk starting at env step T must be predicted from
images captured at step T-d and state from step T. delay=0 == synchronous.
"""
import pytest

from benchmarks.libero.executor import DelayedChunkExecutor


class RecordingPolicy:
    def __init__(self):
        self.calls = []  # (images_step, state_step)

    def __call__(self, images, state, task):
        self.calls.append((images["step"], state))
        base = len(self.calls) * 100
        return [base + i for i in range(5)]


def run_steps(executor, n):
    actions = []
    for t in range(n):
        actions.append(executor.act({"step": t}, t, "task"))
    return actions


def test_sync_delay0_uses_fresh_obs():
    policy = RecordingPolicy()
    ex = DelayedChunkExecutor(policy, k=5, delay=0)
    run_steps(ex, 12)
    # chunks start at steps 0, 5, 10; images and state both from chunk-start step
    assert policy.calls == [(0, 0), (5, 5), (10, 10)]


def test_delay1_stale_images_fresh_state():
    policy = RecordingPolicy()
    ex = DelayedChunkExecutor(policy, k=5, delay=1)
    run_steps(ex, 12)
    # first chunk is a fresh bootstrap; later chunks: images from T-1, state from T
    assert policy.calls == [(0, 0), (4, 5), (9, 10)]


def test_delay3_stale_images_fresh_state():
    policy = RecordingPolicy()
    ex = DelayedChunkExecutor(policy, k=5, delay=3)
    run_steps(ex, 12)
    assert policy.calls == [(0, 0), (2, 5), (7, 10)]


def test_actions_come_from_current_chunk_in_order():
    policy = RecordingPolicy()
    ex = DelayedChunkExecutor(policy, k=5, delay=1)
    actions = run_steps(ex, 10)
    assert actions == [100, 101, 102, 103, 104, 200, 201, 202, 203, 204]


def test_delay_must_be_less_than_k():
    with pytest.raises(AssertionError):
        DelayedChunkExecutor(RecordingPolicy(), k=5, delay=5)


def test_delay1_stale_state_uses_stale_state_too():
    policy = RecordingPolicy()
    ex = DelayedChunkExecutor(policy, k=5, delay=1, stale_state=True)
    run_steps(ex, 12)
    # naive async: both images and state come from the T-1 snapshot
    assert policy.calls == [(0, 0), (4, 4), (9, 9)]


def test_delay3_stale_state_uses_stale_state_too():
    policy = RecordingPolicy()
    ex = DelayedChunkExecutor(policy, k=5, delay=3, stale_state=True)
    run_steps(ex, 12)
    assert policy.calls == [(0, 0), (2, 2), (7, 7)]


def test_delay1_stale_state_false_keeps_fresh_state():
    policy = RecordingPolicy()
    ex = DelayedChunkExecutor(policy, k=5, delay=1, stale_state=False)
    run_steps(ex, 12)
    assert policy.calls == [(0, 0), (4, 5), (9, 10)]
