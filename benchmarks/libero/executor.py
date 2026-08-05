"""Deterministic emulation of VLASH async inference timing for stepped sims.

Mirrors vlash/run.py's VLASHAsyncManager semantics without wall-clock overlap:
the chunk that begins at env step T was requested `delay` steps earlier, so it
sees images from step T-delay but conditions on the true state at step T
(training analogue: use_state_ground_truth=True — stale images, chunk-start
state s_{t+offset}). delay=0 reduces to synchronous inference.

Setting `stale_state=True` emulates a naive async client that has no ground
truth channel: the snapshot taken at the T-delay boundary captures state
alongside images, and the chunk-switch predict call conditions on that stale
state too (fully stale observation), rather than the true state at step T.
"""


class DelayedChunkExecutor:
    def __init__(self, predict_fn, k: int, delay: int, stale_state: bool = False):
        assert 0 <= delay < k, "delay must be in [0, k)"
        self.predict_fn = predict_fn
        self.k = k
        self.delay = delay
        self.stale_state = stale_state
        self.chunk = None
        self.idx = 0
        self.stale_images = None
        self.stale_state_value = None

    def act(self, images: dict, state, task: str):
        if self.chunk is None or self.idx == self.k:
            # First chunk bootstraps with fresh images; afterwards use the
            # snapshot taken `delay` steps before this chunk switch.
            use_images = (
                images
                if (self.stale_images is None or self.delay == 0)
                else self.stale_images
            )
            use_state = (
                state
                if (
                    self.stale_state_value is None
                    or self.delay == 0
                    or not self.stale_state
                )
                else self.stale_state_value
            )
            self.chunk = self.predict_fn(use_images, use_state, task)[: self.k]
            self.idx = 0
        if self.delay > 0 and self.idx == self.k - self.delay:
            self.stale_images = images
            self.stale_state_value = state
        action = self.chunk[self.idx]
        self.idx += 1
        return action


class OverlapChunkExecutor:
    """openpi-cell delay protocol (kinetix eval convention), added for the RTC
    cross-check: request with the CURRENT observation; the new chunk takes
    effect `delay` steps later, so each cycle executes prev_full[k:k+delay]
    (the in-flight overlap) then new[delay:k]. The rtc arm additionally passes
    rtc metadata so the SERVER runs guided inpainting against its cached
    previous chunk (arXiv 2506.07339).

    Differs from DelayedChunkExecutor above (VLASH's native convention:
    stale snapshot at idx==k-delay, full-chunk swap at the boundary): here the
    obs is fresh at request time and the delay is paid in execution overlap.
    Matches openpi examples/libero/main_delay.py so d means the same thing in
    both stacks.
    """

    def __init__(self, predict_fn, k: int, delay: int, arm: str, env_id: int):
        assert arm in ("sync", "naive", "rtc"), arm
        if arm == "sync":
            assert delay == 0, "sync arm is delay-0 by definition"
        assert 0 <= delay <= k, "delay must fit in the execute window"
        self.predict_fn = predict_fn
        self.k = k
        self.delay = delay
        self.arm = arm
        self.env_id = env_id
        self.prev_full = None
        self.buffer = []

    def act(self, images: dict, state, task: str):
        if not self.buffer:
            rtc = None
            if self.arm == "rtc":
                rtc = {"env_id": self.env_id, "delay": self.delay, "executed": self.k}
            full = self.predict_fn(images, state, task, rtc=rtc, full=True)
            assert len(full) >= self.k + self.delay, (
                f"chunk horizon {len(full)} too short for k={self.k} d={self.delay}"
            )
            if self.prev_full is None or self.delay == 0:
                committed = full[: self.k]
            else:
                import numpy as _np

                committed = _np.concatenate(
                    [self.prev_full[self.k : self.k + self.delay], full[self.delay : self.k]]
                )
            self.prev_full = full
            self.buffer = list(committed)
        return self.buffer.pop(0)
