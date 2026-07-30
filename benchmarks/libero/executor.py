"""Deterministic emulation of VLASH async inference timing for stepped sims.

Mirrors vlash/run.py's VLASHAsyncManager semantics without wall-clock overlap:
the chunk that begins at env step T was requested `delay` steps earlier, so it
sees images from step T-delay but conditions on the true state at step T
(training analogue: use_state_ground_truth=True — stale images, chunk-start
state s_{t+offset}). delay=0 reduces to synchronous inference.
"""


class DelayedChunkExecutor:
    def __init__(self, predict_fn, k: int, delay: int):
        assert 0 <= delay < k, "delay must be in [0, k)"
        self.predict_fn = predict_fn
        self.k = k
        self.delay = delay
        self.chunk = None
        self.idx = 0
        self.stale_images = None

    def act(self, images: dict, state, task: str):
        if self.chunk is None or self.idx == self.k:
            # First chunk bootstraps with fresh images; afterwards use the
            # snapshot taken `delay` steps before this chunk switch.
            use_images = (
                images
                if (self.stale_images is None or self.delay == 0)
                else self.stale_images
            )
            self.chunk = self.predict_fn(use_images, state, task)[: self.k]
            self.idx = 0
        if self.delay > 0 and self.idx == self.k - self.delay:
            self.stale_images = images
        action = self.chunk[self.idx]
        self.idx += 1
        return action
