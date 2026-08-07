# LIBERO Success-Rate Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce VLASH's LIBERO success-rate table for π0.5 (paper: sync/delay=0 → 96.8%, VLASH delay=1 → 97.2%, VLASH delay=3 → 94.6%) by fine-tuning `lerobot/pi05_base` with VLASH async training on a LIBERO LeRobot dataset and evaluating in the LIBERO simulator with emulated inference delay, on nano4.

**Architecture:** Two-process eval — a Flask policy server in the vlash conda env (GPU) and a LIBERO sim client in the existing `/work/roboleon1295/.venv-libero-vanilla` venv (they have incompatible deps), talking npz-over-HTTP on localhost inside one Slurm job. Delay is emulated deterministically: the chunk that starts executing at env step T is predicted from images captured at step T−d and the true state at step T — exactly matching training with `use_state_ground_truth=True` (stale images, chunk-start state). Training runs on 4×H200 via the existing `vlash train` CLI; one small vlash code change is needed (`use_state_ground_truth` is not currently plumbed from config to dataset, and LIBERO requires it because state_dim=8 ≠ action_dim=7).

**Tech Stack:** vlash @ branch `libero-eval` (fork of mit-han-lab/vlash), lerobot 0.4.1, Flask (already a vlash dep), LIBERO 0.1.0 + robosuite 1.4.1 + mujoco 3.2.3 (existing venv), Slurm on nano4, wandb.

## Global Constraints

- **Code reaches nano4 only via GitHub** (push branch to fork `Chung-I/vlash`, pull on nano4). Never scp/rsync source.
- All nano4 installs/caches/outputs under `/work/roboleon1295/`, never `/home/roboleon1295`.
- Every nano4 GPU job: `#SBATCH --account=MST114563`, ≤12 CPUs + ≤200 GB per GPU, honest `--time`.
- vlash env on nano4 requires `export LD_LIBRARY_PATH=/work/roboleon1295/envs/vlash/lib:$LD_LIBRARY_PATH` (GLIBCXX) and `module load miniconda3/26.1.1; source activate /work/roboleon1295/envs/vlash`.
- Slurm jobs: `export HOME=/work/roboleon1295/jobhome` (wandb `.netrc` and caches live there) and `HF_HOME=/work/roboleon1295/.cache/huggingface`.
- LIBERO client env: `/work/roboleon1295/.venv-libero-vanilla` with `MUJOCO_GL=egl`.
- All experiment results MUST be logged to wandb (project `vlash-libero`).
- Paper-matching hyperparameters: 30K steps, effective batch 32 (8/GPU × 4 GPUs), `max_delay_steps=3`, `state_cond=true` (already in `examples/train/pi05/async.yaml`), execution horizon K=5, 50 trials/task, delays {0, 1, 3}.

---

### Task 0: Fork, branch, and commit this plan

**Files:**
- Create: `docs/superpowers/plans/2026-07-30-libero-success-rate-repro.md` (this file)

**Interfaces:**
- Produces: branch `libero-eval` on `https://github.com/Chung-I/vlash` that every later task commits to; nano4 checkout `/work/roboleon1295/vlash` tracking it.

- [ ] **Step 1: Create fork and branch locally**

```bash
cd /home/chungyili/Codes/vlash
gh repo fork mit-han-lab/vlash --remote=false --clone=false   # no-op if fork exists
git remote add fork https://github.com/Chung-I/vlash.git 2>/dev/null || true
git checkout -b libero-eval
```

- [ ] **Step 2: Commit the plan and push**

```bash
git add docs/superpowers/plans/2026-07-30-libero-success-rate-repro.md
git commit -m "docs: LIBERO success-rate reproduction plan"
git push -u fork libero-eval
```

- [ ] **Step 3: Point nano4's checkout at the branch**

```bash
ssh nano4 'cd /work/roboleon1295/vlash && git remote add fork https://github.com/Chung-I/vlash.git 2>/dev/null; git fetch fork && git checkout libero-eval && git log --oneline -1'
```

Expected: the plan commit hash, same as local.

---

### Task 1: Plumb `use_state_ground_truth` from config to datasets

**Files:**
- Modify: `vlash/configs/train_config.py` (after the `shared_observation` field, ~line 103)
- Modify: `vlash/train.py` (both dataset constructor calls, ~lines 120–150)
- Test: `tests/test_state_ground_truth_wiring.py` (new; repo has no tests dir yet — create it)

**Interfaces:**
- Consumes: `VLASHDataset(..., use_state_ground_truth: bool = False)` and `SharedObservationVLASHDataset(..., use_state_ground_truth: bool = False)` — already implemented in `vlash/datasets/vlash_dataset.py`.
- Produces: CLI flag `--use_state_ground_truth=true` on `vlash train`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state_ground_truth_wiring.py
"""use_state_ground_truth must flow from TrainConfig into both dataset classes."""
import inspect
from unittest.mock import patch

from vlash.configs.train_config import TrainConfig
import vlash.train as train_mod


def test_config_has_field():
    assert "use_state_ground_truth" in {f.name for f in TrainConfig.__dataclass_fields__.values()}
    assert TrainConfig.__dataclass_fields__["use_state_ground_truth"].default is False


def test_make_dataset_passes_flag_shared():
    src = inspect.getsource(train_mod)
    # Both constructor call sites must forward the flag
    assert src.count("use_state_ground_truth=cfg.use_state_ground_truth") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_state_ground_truth_wiring.py -v`
Expected: FAIL (`use_state_ground_truth` not in fields; count == 0)

- [ ] **Step 3: Implement**

In `vlash/configs/train_config.py`, directly after the `shared_observation: bool = False` field:

```python
    # Use recorded future state s_{t+offset} for the async-offset condition
    # instead of the previous-action proxy. REQUIRED when state_dim != action_dim
    # (e.g. LIBERO: 8-dim state, 7-dim OSC-delta action).
    use_state_ground_truth: bool = False
```

In `vlash/train.py`, add to BOTH the `SharedObservationVLASHDataset(...)` and `VLASHDataset(...)` constructor calls, after `max_delay_steps=cfg.max_delay_steps,`:

```python
            use_state_ground_truth=cfg.use_state_ground_truth,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_state_ground_truth_wiring.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add vlash/configs/train_config.py vlash/train.py tests/test_state_ground_truth_wiring.py
git commit -m "feat(train): expose use_state_ground_truth for state_dim != action_dim datasets (LIBERO)"
```

---

### Task 2: Probe and pin the LIBERO training dataset

**Files:**
- Create: `benchmarks/libero/__init__.py` (empty)
- Create: `benchmarks/libero/libero_config.py`
- Create: `benchmarks/libero/probe_dataset.py`

**Interfaces:**
- Produces: `libero_config.py` constants used by every later task: `DATASET_REPO_ID: str`, `IMAGE_KEYS: dict[str, str]` (client array name → policy feature key), `STATE_DIM=8`, `ACTION_DIM=7`, `K=5`, `DELAYS=[0,1,3]`, `SUITES: dict[str, int]` (suite → max env steps), `NUM_TRIALS=50`, `SETTLE_STEPS=10`, `SERVER_PORT=5901`.

- [ ] **Step 1: Write the probe script**

```python
# benchmarks/libero/probe_dataset.py
"""Validate a candidate LIBERO LeRobot dataset against lerobot 0.4.1.

Run (nano4 login node, vlash env):
    python benchmarks/libero/probe_dataset.py HuggingFaceVLA/libero
    python benchmarks/libero/probe_dataset.py physical-intelligence/libero
Prints camera keys, shapes, fps, episode/frame counts. Exit 0 = usable.
"""
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def main(repo_id: str) -> None:
    meta = LeRobotDatasetMetadata(repo_id)
    print("repo_id:", repo_id)
    print("total_episodes:", meta.total_episodes, "total_frames:", meta.total_frames)
    print("fps:", meta.fps)
    print("camera_keys:", meta.camera_keys)
    state = meta.features["observation.state"]["shape"]
    action = meta.features["action"]["shape"]
    print("state shape:", state, "action shape:", action)
    assert state == (8,), f"expected 8-dim state, got {state}"
    assert action == (7,), f"expected 7-dim action, got {action}"
    assert len(meta.camera_keys) == 2, f"expected 2 cameras, got {meta.camera_keys}"
    print("PROBE_OK")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Write the config with the primary candidate**

```python
# benchmarks/libero/libero_config.py
"""Constants shared by the LIBERO training and evaluation pipeline.

DATASET_REPO_ID / IMAGE_KEYS are confirmed by probe_dataset.py (Task 2).
IMAGE_KEYS maps npz array names sent by the sim client to the policy's
image feature keys (which come from the training dataset's camera keys).
"""
DATASET_REPO_ID = "HuggingFaceVLA/libero"  # fallback: physical-intelligence/libero

IMAGE_KEYS = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
}

STATE_DIM = 8   # eef_pos(3) + eef axis-angle(3) + gripper_qpos(2)
ACTION_DIM = 7  # OSC delta pose (6) + gripper (1)

K = 5           # execution horizon (paper: K=5)
DELAYS = [0, 1, 3]

# suite name -> max env steps per episode (openpi LIBERO conventions)
SUITES = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
NUM_TRIALS = 50
SETTLE_STEPS = 10  # dummy steps after set_init_state for physics to settle
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

SERVER_PORT = 5901
```

- [ ] **Step 3: Commit, push, run the probe on nano4**

```bash
git add benchmarks/libero/
git commit -m "feat(libero): dataset probe + pinned eval constants"
git push fork libero-eval
ssh nano4 'cd /work/roboleon1295/vlash && git pull fork libero-eval &&
  module load miniconda3/26.1.1 && source activate /work/roboleon1295/envs/vlash &&
  export LD_LIBRARY_PATH=/work/roboleon1295/envs/vlash/lib:$LD_LIBRARY_PATH HF_HOME=/work/roboleon1295/.cache/huggingface &&
  python benchmarks/libero/probe_dataset.py HuggingFaceVLA/libero'
```

Expected: `PROBE_OK` with 2 camera keys. **If it fails** (missing repo, wrong dims, or v-format rejection), run the probe on `physical-intelligence/libero`; whichever passes becomes `DATASET_REPO_ID`, and `IMAGE_KEYS` values are updated to that dataset's `camera_keys` exactly as printed. Commit any correction:

```bash
git add benchmarks/libero/libero_config.py
git commit -m "fix(libero): pin dataset repo_id/camera keys per probe" && git push fork libero-eval
```

- [ ] **Step 4: Pre-download the dataset into the /work HF cache (login node, background)**

```bash
ssh nano4 'export HF_HOME=/work/roboleon1295/.cache/huggingface &&
  nohup hf download HuggingFaceVLA/libero --repo-type dataset \
    > /work/roboleon1295/stage-latency/dl_libero_lerobot.log 2>&1 & echo started'
```

Verify later: `tail -1 dl_libero_lerobot.log` shows the snapshot path.

---

### Task 3: DelayedChunkExecutor (pure logic + tests, local)

**Files:**
- Create: `benchmarks/libero/executor.py`
- Test: `tests/test_delayed_chunk_executor.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `DelayedChunkExecutor(predict_fn, k: int, delay: int)` with method `act(images: dict, state, task: str) -> action`. `predict_fn(images: dict, state, task: str) -> sequence of >= k actions`. Used by the eval client (Task 5).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_delayed_chunk_executor.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_delayed_chunk_executor.py -v`
Expected: FAIL with `ModuleNotFoundError: benchmarks.libero.executor`

- [ ] **Step 3: Implement**

```python
# benchmarks/libero/executor.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_delayed_chunk_executor.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add benchmarks/libero/executor.py tests/test_delayed_chunk_executor.py
git commit -m "feat(libero): delayed chunk executor emulating async timing"
```

---

### Task 4: Policy server (vlash env) + local round-trip test

**Files:**
- Create: `benchmarks/libero/serve_policy.py`
- Test: `tests/test_serve_policy_roundtrip.py`

**Interfaces:**
- Consumes: `IMAGE_KEYS`, `STATE_DIM`, `K`, `SERVER_PORT` from `benchmarks.libero.libero_config`.
- Produces: HTTP API used by Task 5's client:
  - `GET /health` → `{"status": "ok"}`
  - `POST /predict` with npz body containing arrays `image` (H,W,3 uint8), `wrist_image` (H,W,3 uint8), `state` (8, float32), `task` (0-d unicode) → npz response with `actions` (K, 7 float32)
- Also produces `load_policy(checkpoint: str | None) -> policy` and `make_app(policy) -> Flask` for testing.

- [ ] **Step 1: Write the failing round-trip test (random-weight policy, local 5090)**

```python
# tests/test_serve_policy_roundtrip.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_serve_policy_roundtrip.py -v`
Expected: FAIL with `ModuleNotFoundError: benchmarks.libero.serve_policy`

- [ ] **Step 3: Implement the server**

```python
# benchmarks/libero/serve_policy.py
"""Flask policy server for LIBERO evaluation.

Runs in the vlash conda env (GPU). The LIBERO sim client (separate venv)
posts observations as npz bytes and receives the first K actions of the
predicted chunk. Latency does not matter here (delay is emulated by the
client), so the model is NOT torch.compiled.

Usage:
    python benchmarks/libero/serve_policy.py --checkpoint /path/to/pretrained_model [--port 5901]
"""
import argparse
import io
import logging

import numpy as np
import torch
from flask import Flask, request

from benchmarks.libero.libero_config import IMAGE_KEYS, K, SERVER_PORT
from vlash.policies.factory import get_policy_class


def load_policy(checkpoint: str):
    policy_cls = get_policy_class("pi05")
    policy = policy_cls.from_pretrained(pretrained_name_or_path=checkpoint)
    policy.config.compile_model = False
    return policy.to(policy.config.device).eval()


def make_app(policy) -> Flask:
    app = Flask("vlash_libero_server")
    device = next(policy.parameters()).device

    @app.route("/health")
    def health():
        return {"status": "ok"}

    @app.route("/predict", methods=["POST"])
    def predict():
        data = np.load(io.BytesIO(request.data), allow_pickle=False)
        batch = {}
        for arr_name, feature_key in IMAGE_KEYS.items():
            img = data[arr_name]  # (H, W, 3) uint8
            t = torch.from_numpy(img).to(device).permute(2, 0, 1).float() / 255.0
            batch[feature_key] = t.unsqueeze(0)
        batch["observation.state"] = (
            torch.from_numpy(np.asarray(data["state"], dtype=np.float32))
            .unsqueeze(0)
            .to(device)
        )
        batch["task"] = [str(data["task"])]
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)
        actions = chunk[0, :K].float().cpu().numpy().astype(np.float32)
        buf = io.BytesIO()
        np.savez(buf, actions=actions)
        return buf.getvalue()

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    app = make_app(load_policy(args.checkpoint))
    app.run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_serve_policy_roundtrip.py -v`
Expected: 2 passed (~1–2 min: builds a random pi05 on the 5090)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/libero/serve_policy.py tests/test_serve_policy_roundtrip.py
git commit -m "feat(libero): flask policy server with npz predict API"
```

---

### Task 5: LIBERO eval client (obs conversion + episode loop)

**Files:**
- Create: `benchmarks/libero/eval_client.py`
- Test: `tests/test_eval_client_obs.py`

**Interfaces:**
- Consumes: `DelayedChunkExecutor` (Task 3); server HTTP API (Task 4); `libero_config` constants.
- Produces: CLI `python benchmarks/libero/eval_client.py --suite libero_spatial --delay 1 --out results.json [--num-trials 50] [--port 5901]`; writes JSON `{"suite", "delay", "per_task": {task_name: {"successes": int, "trials": int}}, "success_rate": float}`.
- NOTE: this file runs in the **libero venv** (no vlash/lerobot/torch imports at module top level — only numpy, requests, and LIBERO). `DelayedChunkExecutor` is imported from `benchmarks.libero.executor`, which is dependency-free.

- [ ] **Step 1: Write the failing tests for obs conversion (pure numpy, run locally)**

```python
# tests/test_eval_client_obs.py
"""Observation conversion: LIBERO raw obs -> policy inputs.

Conventions (openpi LIBERO): both camera images are rendered upside-down and
must be rotated 180 degrees; state is eef_pos(3) + axis-angle(3) + gripper(2).
"""
import numpy as np

from benchmarks.libero.eval_client import libero_obs_to_arrays, quat2axisangle


def test_images_rotated_180_and_contiguous():
    obs = {
        "agentview_image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "robot0_eye_in_hand_image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    arrays = libero_obs_to_arrays(obs)
    np.testing.assert_array_equal(
        arrays["image"], obs["agentview_image"][::-1, ::-1]
    )
    assert arrays["image"].flags["C_CONTIGUOUS"]
    assert arrays["state"].shape == (8,)
    assert arrays["state"].dtype == np.float32


def test_quat2axisangle_identity():
    np.testing.assert_allclose(
        quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0])), np.zeros(3), atol=1e-7
    )


def test_quat2axisangle_90deg_z():
    s = np.sin(np.pi / 4)
    aa = quat2axisangle(np.array([0.0, 0.0, s, np.cos(np.pi / 4)]))
    np.testing.assert_allclose(aa, [0.0, 0.0, np.pi / 2], atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_client_obs.py -v`
Expected: FAIL with `ModuleNotFoundError: benchmarks.libero.eval_client`

- [ ] **Step 3: Implement the client**

```python
# benchmarks/libero/eval_client.py
"""LIBERO simulator evaluation client (runs in the LIBERO venv, CPU mujoco).

Drives LIBERO episodes, converts observations, queries the vlash policy
server over HTTP, and executes chunks under emulated inference delay.

Usage (inside a Slurm job, after the server reports healthy):
    MUJOCO_GL=egl python benchmarks/libero/eval_client.py \
        --suite libero_spatial --delay 1 --out /work/.../spatial_d1.json
"""
import argparse
import io
import json
import math
import pathlib
import sys

import numpy as np
import requests

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from executor import DelayedChunkExecutor  # noqa: E402
from libero_config import (  # noqa: E402
    DUMMY_ACTION,
    K,
    NUM_TRIALS,
    SERVER_PORT,
    SETTLE_STEPS,
    SUITES,
)


def quat2axisangle(quat):
    """xyzw quaternion -> axis-angle (matches robosuite/openpi convention)."""
    quat = np.asarray(quat, dtype=np.float64)
    w = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(w)) / den


def libero_obs_to_arrays(obs) -> dict:
    """LIBERO raw obs dict -> named arrays for the policy server."""
    return {
        "image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "state": np.concatenate(
            [
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ]
        ).astype(np.float32),
    }


class RemotePolicy:
    def __init__(self, port: int, task: str):
        self.url = f"http://127.0.0.1:{port}/predict"
        self.task = task

    def __call__(self, images: dict, state, task: str):
        buf = io.BytesIO()
        np.savez(
            buf,
            image=images["image"],
            wrist_image=images["wrist_image"],
            state=np.asarray(state, dtype=np.float32),
            task=np.array(task),
        )
        resp = requests.post(self.url, data=buf.getvalue(), timeout=120)
        resp.raise_for_status()
        return np.load(io.BytesIO(resp.content))["actions"]


def run_episode(env, init_state, task_str: str, delay: int, max_steps: int, port: int) -> bool:
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    policy = RemotePolicy(port, task_str)
    executor = DelayedChunkExecutor(policy, k=K, delay=delay)

    for _ in range(max_steps):
        arrays = libero_obs_to_arrays(obs)
        images = {"image": arrays["image"], "wrist_image": arrays["wrist_image"]}
        action = executor.act(images, arrays["state"], task_str)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True
    return False


def main():
    from libero.libero import benchmark as libero_benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=list(SUITES))
    parser.add_argument("--delay", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    args = parser.parse_args()

    suite = libero_benchmark.get_benchmark_dict()[args.suite]()
    max_steps = SUITES[args.suite]
    results = {}

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        task_str = task.language
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        env.seed(0)
        successes = 0
        for ep in range(args.num_trials):
            ok = run_episode(
                env, init_states[ep], task_str, args.delay, max_steps, args.port
            )
            successes += int(ok)
            print(
                f"[{args.suite} d={args.delay}] task {task_id} ({task_str[:40]}) "
                f"ep {ep}: {'OK' if ok else 'fail'} ({successes}/{ep + 1})",
                flush=True,
            )
        env.close()
        results[task.name] = {"successes": successes, "trials": args.num_trials}

    total_s = sum(r["successes"] for r in results.values())
    total_t = sum(r["trials"] for r in results.values())
    out = {
        "suite": args.suite,
        "delay": args.delay,
        "per_task": results,
        "success_rate": total_s / total_t,
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print("WROTE", args.out, "success_rate", out["success_rate"], flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the conversion tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_client_obs.py -v`
Expected: 3 passed (module-level imports are numpy/requests only, so this works in the vlash venv; `requests` is a transitive dep — if missing, `uv pip install -p .venv/bin/python requests`)

- [ ] **Step 5: Commit and push**

```bash
git add benchmarks/libero/eval_client.py tests/test_eval_client_obs.py
git commit -m "feat(libero): sim eval client with delayed-chunk execution"
git push fork libero-eval
```

---

### Task 6: nano4 plumbing smoke test (random-weight policy, 2 episodes)

**Files:**
- Create: `benchmarks/libero/smoke_server.py`
- Create: `benchmarks/libero/smoke.sbatch`

**Interfaces:**
- Consumes: server/client/executor from Tasks 3–5; libero venv; vlash env.
- Produces: proof that the two-env HTTP pipeline runs LIBERO end-to-end on a GPU node (success rate irrelevant — random weights).

- [ ] **Step 1: Write a server entry that builds a random LIBERO-shaped policy**

```python
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
```

- [ ] **Step 2: Write the smoke sbatch**

```bash
# benchmarks/libero/smoke.sbatch
#!/bin/bash
#SBATCH --job-name=libero-smoke
#SBATCH --partition=dev
#SBATCH --account=MST114563
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/work/roboleon1295/vlash-eval/smoke-%j.log

set -x
mkdir -p /work/roboleon1295/vlash-eval
module purge; module load miniconda3/26.1.1
source activate /work/roboleon1295/envs/vlash
export LD_LIBRARY_PATH=/work/roboleon1295/envs/vlash/lib:${LD_LIBRARY_PATH:-}
export HOME=/work/roboleon1295/jobhome HF_HOME=/work/roboleon1295/.cache/huggingface
cd /work/roboleon1295/vlash

python benchmarks/libero/smoke_server.py > /work/roboleon1295/vlash-eval/smoke-server-$SLURM_JOB_ID.log 2>&1 &
SERVER_PID=$!
until curl -s http://127.0.0.1:5901/health | grep -q ok; do sleep 5; done

MUJOCO_GL=egl /work/roboleon1295/.venv-libero-vanilla/bin/python \
  benchmarks/libero/eval_client.py --suite libero_spatial --delay 1 \
  --num-trials 2 --out /work/roboleon1295/vlash-eval/smoke.json
kill $SERVER_PID
cat /work/roboleon1295/vlash-eval/smoke.json
```

- [ ] **Step 3: Commit, push, pull on nano4, submit**

```bash
git add benchmarks/libero/smoke_server.py benchmarks/libero/smoke.sbatch
git commit -m "feat(libero): plumbing smoke test (random weights, 2 episodes/task)"
git push fork libero-eval
ssh nano4 'cd /work/roboleon1295/vlash && git pull fork libero-eval &&
  /work/roboleon1295/.venv-libero-vanilla/bin/python -c "import requests" 2>/dev/null \
    || /work/roboleon1295/.venv-libero-vanilla/bin/pip install requests &&
  sbatch benchmarks/libero/smoke.sbatch'
```

- [ ] **Step 4: Verify**

Watch `/work/roboleon1295/vlash-eval/smoke-<job>.log` until `WROTE ... smoke.json` appears (2 episodes × 10 tasks, random policy → success_rate ≈ 0.0 is EXPECTED and fine). Failure modes to fix here, not later: EGL rendering, camera key mismatch, npz encoding, LIBERO init-state shapes.

---

### Task 7: Launch VLASH async fine-tuning (4×H200)

**Files:**
- Create: `benchmarks/libero/train_libero.sbatch`

**Interfaces:**
- Consumes: Task 1's `--use_state_ground_truth` flag; pinned `DATASET_REPO_ID`; `lerobot/pi05_base` (auto-downloaded to `HF_HOME`).
- Produces: checkpoint dir `/work/roboleon1295/vlash/outputs/train/async_libero/checkpoints/030000/pretrained_model` consumed by Task 8; wandb run in project `vlash-libero`.

- [ ] **Step 1: Write the training sbatch**

```bash
# benchmarks/libero/train_libero.sbatch
#!/bin/bash
#SBATCH --job-name=vlash-libero-train
#SBATCH --partition=8gpus
#SBATCH --account=MST114563
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=/work/roboleon1295/vlash-eval/train-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leon129506@gmail.com

set -x
module purge; module load miniconda3/26.1.1
source activate /work/roboleon1295/envs/vlash
export LD_LIBRARY_PATH=/work/roboleon1295/envs/vlash/lib:${LD_LIBRARY_PATH:-}
export HOME=/work/roboleon1295/jobhome
mkdir -p "$HOME/.triton" "$HOME/.cache"
export TRITON_CACHE_DIR="$HOME/.triton" XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME=/work/roboleon1295/.cache/huggingface
cd /work/roboleon1295/vlash

srun vlash train examples/train/pi05/async.yaml HuggingFaceVLA/libero \
  --output_dir=outputs/train/async_libero \
  --job_name=async_libero \
  --steps=30000 \
  --batch_size=8 \
  --max_delay_steps=3 \
  --use_state_ground_truth=true \
  --num_workers=8 \
  --save_freq=10000 \
  --scheduler.num_decay_steps=30000 \
  --wandb.project=vlash-libero
```

(Effective batch = 8 × 4 GPUs = 32, matching the paper. `vlash train` auto-detects 4 GPUs and uses `accelerate launch --multi_gpu`. `state_cond=true` and `shared_observation=true` come from `async.yaml`. If the pinned dataset changed in Task 2, use that repo_id here.)

- [ ] **Step 2: Commit, push, pull, submit**

```bash
git add benchmarks/libero/train_libero.sbatch
git commit -m "feat(libero): 4-GPU async fine-tuning job (paper hparams)"
git push fork libero-eval
ssh nano4 'cd /work/roboleon1295/vlash && git pull fork libero-eval && sbatch benchmarks/libero/train_libero.sbatch'
```

- [ ] **Step 3: Verify training is healthy (first 30 min)**

- wandb run appears in project `vlash-libero` with decreasing `loss`.
- Log shows `Creating SharedObservationVLASHDataset with max_delay_steps=3` and NO `Unsupported state_dim != action_dim` error (that error means Task 1's flag isn't active — stop and fix).
- Throughput sanity: paper reports ~129 ms/step on 4×H100 at batch 64 with shared observation; expect a similar order on 4×H200 at batch 32 (≲2 h for 30K steps + dataloader overhead). If projected wall time exceeds the 12 h limit, investigate the dataloader (raise `--num_workers`) before resubmitting.

---

### Task 8: Full evaluation sweep (4 suites × 3 delays)

**Files:**
- Create: `benchmarks/libero/eval_array.sbatch`

**Interfaces:**
- Consumes: trained checkpoint from Task 7; server (Task 4); client (Task 5).
- Produces: 12 JSON files `/work/roboleon1295/vlash-eval/results/<suite>_d<delay>.json` consumed by Task 9.

- [ ] **Step 1: Write the array job**

```bash
# benchmarks/libero/eval_array.sbatch
#!/bin/bash
#SBATCH --job-name=vlash-libero-eval
#SBATCH --partition=8gpus
#SBATCH --account=MST114563
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --array=0-11%8
#SBATCH --output=/work/roboleon1295/vlash-eval/eval-%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leon129506@gmail.com

set -x
SUITES=(libero_spatial libero_object libero_goal libero_10)
DELAYS=(0 1 3)
SUITE=${SUITES[$((SLURM_ARRAY_TASK_ID / 3))]}
DELAY=${DELAYS[$((SLURM_ARRAY_TASK_ID % 3))]}
PORT=$((5901 + SLURM_ARRAY_TASK_ID))
CKPT=/work/roboleon1295/vlash/outputs/train/async_libero/checkpoints/030000/pretrained_model
OUT=/work/roboleon1295/vlash-eval/results
mkdir -p $OUT

module purge; module load miniconda3/26.1.1
source activate /work/roboleon1295/envs/vlash
export LD_LIBRARY_PATH=/work/roboleon1295/envs/vlash/lib:${LD_LIBRARY_PATH:-}
export HOME=/work/roboleon1295/jobhome HF_HOME=/work/roboleon1295/.cache/huggingface
cd /work/roboleon1295/vlash

python benchmarks/libero/serve_policy.py --checkpoint $CKPT --port $PORT \
  > /work/roboleon1295/vlash-eval/server-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
SERVER_PID=$!
until curl -s http://127.0.0.1:$PORT/health | grep -q ok; do
  kill -0 $SERVER_PID || { echo "SERVER DIED"; exit 1; }
  sleep 5
done

MUJOCO_GL=egl /work/roboleon1295/.venv-libero-vanilla/bin/python \
  benchmarks/libero/eval_client.py --suite $SUITE --delay $DELAY --port $PORT \
  --out $OUT/${SUITE}_d${DELAY}.json
kill $SERVER_PID
```

- [ ] **Step 2: Commit, push, pull, submit (after Task 7's checkpoint exists)**

```bash
git add benchmarks/libero/eval_array.sbatch
git commit -m "feat(libero): 12-shard eval array (4 suites x 3 delays)"
git push fork libero-eval
ssh nano4 'ls /work/roboleon1295/vlash/outputs/train/async_libero/checkpoints/030000/pretrained_model &&
  cd /work/roboleon1295/vlash && git pull fork libero-eval && sbatch benchmarks/libero/eval_array.sbatch'
```

- [ ] **Step 3: Monitor**

Each shard prints one line per episode. Expected wall time ~6–10 h per shard (500 episodes × ~1 min). Spot-check an early shard: if `libero_spatial_d0` is trending far below ~90% after several tasks, stop the array and debug (normalization stats, camera mapping, action convention) rather than burning 12 shards.

---

### Task 9: Aggregate, log to wandb, report vs paper

**Files:**
- Create: `benchmarks/libero/aggregate_results.py`

**Interfaces:**
- Consumes: the 12 result JSONs.
- Produces: wandb run `libero-success-rates` in project `vlash-libero`; printed comparison table.

- [ ] **Step 1: Write the aggregator**

```python
# benchmarks/libero/aggregate_results.py
"""Aggregate LIBERO eval shards and log the success-rate table to wandb.

Usage: python benchmarks/libero/aggregate_results.py /work/roboleon1295/vlash-eval/results
"""
import json
import pathlib
import sys

import wandb

PAPER = {0: 0.968, 1: 0.972, 3: 0.946}
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DELAYS = [0, 1, 3]


def main(results_dir: str) -> None:
    root = pathlib.Path(results_dir)
    table = {}
    for delay in DELAYS:
        rates = []
        for suite in SUITES:
            data = json.loads((root / f"{suite}_d{delay}.json").read_text())
            table[f"{suite}/d{delay}"] = data["success_rate"]
            rates.append(data["success_rate"])
        table[f"avg/d{delay}"] = sum(rates) / len(rates)

    run = wandb.init(project="vlash-libero", name="libero-success-rates", job_type="eval")
    wandb.log(table)
    print(f"{'':>16}" + "".join(f"  delay={d}" for d in DELAYS))
    for suite in SUITES:
        print(f"{suite:>16}" + "".join(f"  {table[f'{suite}/d{d}']:.3f}" for d in DELAYS))
    print(f"{'AVG (ours)':>16}" + "".join(f"  {table[f'avg/d{d}']:.3f}" for d in DELAYS))
    print(f"{'paper':>16}" + "".join(f"  {PAPER[d]:.3f}" for d in DELAYS))
    run.finish()


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Run on nano4 once all 12 JSONs exist**

```bash
ssh nano4 'cd /work/roboleon1295/vlash &&
  module load miniconda3/26.1.1 && source activate /work/roboleon1295/envs/vlash &&
  export LD_LIBRARY_PATH=/work/roboleon1295/envs/vlash/lib:$LD_LIBRARY_PATH HOME=/work/roboleon1295/jobhome &&
  python benchmarks/libero/aggregate_results.py /work/roboleon1295/vlash-eval/results'
```

- [ ] **Step 3: Commit and push**

```bash
git add benchmarks/libero/aggregate_results.py
git commit -m "feat(libero): result aggregation + wandb table"
git push fork libero-eval
```

- [ ] **Step 4: Acceptance check**

Reproduction succeeds if the delay-averaged numbers land within a few points of the paper (96.8 / 97.2 / 94.6) and, critically, the *shape* holds: delay=1 ≈ delay=0 (no async penalty) and delay=3 within ~3 points. If sync (d=0) is far below 96.8%, the problem is training/eval fidelity (dataset choice, normalization, conventions), not the async mechanism — debug there first with the systematic-debugging skill.

---

## Self-Review Notes

- **Spec coverage:** train (Task 7), delay-emulated eval for d∈{0,1,3} (Tasks 3/5/8), the state_dim≠action_dim blocker (Task 1), dataset uncertainty isolated to one probe (Task 2), plumbing de-risked before GPU-hours are spent (Tasks 4/6), wandb rule (Tasks 7/9), GitHub-only code sync (every task pushes; nano4 pulls).
- **Known risks called out in-plan:** dataset format compatibility (Task 2 fallback), LIBERO obs conventions (tested in Task 5, exercised in Task 6), early-abort criterion for a bad eval sweep (Task 8 Step 3).
- **Type consistency:** `predict_fn(images: dict, state, task) -> array[(≥K),7]` is uniform across executor/RemotePolicy/server; npz array names `image`/`wrist_image`/`state`/`task` uniform across client/server/tests; `IMAGE_KEYS` maps those names to policy feature keys in exactly one place (`libero_config.py`).
