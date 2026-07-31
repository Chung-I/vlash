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
import os
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

    import torch
    _orig_torch_load = torch.load

    def _load_full(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    # LIBERO's task init states are pickled package data (trusted, ships with
    # the libero package); torch>=2.6 defaults weights_only=True and rejects them.
    torch.load = _load_full

    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=list(SUITES))
    parser.add_argument("--delay", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    args = parser.parse_args()

    suite = libero_benchmark.get_benchmark_dict()[args.suite]()
    max_steps = SUITES[args.suite]

    out_path = pathlib.Path(args.out)
    results = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            results = prev.get("per_task", {})
        except (json.JSONDecodeError, OSError):
            results = {}

    def write_results():
        total_s = sum(r["successes"] for r in results.values())
        total_t = sum(r["trials"] for r in results.values())
        out = {
            "suite": args.suite,
            "delay": args.delay,
            "per_task": results,
            "success_rate": (total_s / total_t) if total_t else 0.0,
        }
        tmp_path = pathlib.Path(str(out_path) + ".tmp")
        tmp_path.write_text(json.dumps(out, indent=2))
        os.replace(tmp_path, out_path)
        return out

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        task_str = task.language
        if task.name in results:
            print(
                f"[{args.suite} d={args.delay}] task {task_id} ({task_str[:40]}) "
                "SKIP (already done, resuming)",
                flush=True,
            )
            continue
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        # Benchmark-SAFE OffScreenRenderEnv kwargs: hard_reset=True rebuilds
        # the sim on every reset (avoids a corrupt-context re-reset path that
        # SIGABRTs under EGL on the second+ episode); explicit camera_names
        # and render_gpu_device_id pin the offscreen rendering pipeline.
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=256,
            camera_widths=256,
            camera_names=["agentview", "robot0_eye_in_hand"],
            control_freq=20,
            hard_reset=True,
            render_gpu_device_id=0,
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
        write_results()

    out = write_results()
    print("WROTE", args.out, "success_rate", out["success_rate"], flush=True)


if __name__ == "__main__":
    main()
