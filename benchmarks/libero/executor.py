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
