"""LIBERO simulator evaluation client (runs in the LIBERO venv, CPU mujoco).

Drives LIBERO episodes, converts observations, queries the vlash policy
server over HTTP, and executes chunks under emulated inference delay.

Each LIBERO task's episodes run in a fresh subprocess (multiprocessing
"spawn" context): long-horizon scenes (libero_10, 520 steps/episode) hit a
stochastic native abort (SIGABRT) somewhere in the MuJoCo/EGL/robosuite
render stack that a single long-lived process cannot recover from. Spawning
one process per task attempt gives each attempt a clean EGL/MuJoCo context;
if the worker dies mid-task the parent respawns it starting at the next
unfinished episode, so progress is never lost below episode granularity.

Usage (inside a Slurm job, after the server reports healthy):
    MUJOCO_GL=egl python benchmarks/libero/eval_client.py \
        --suite libero_spatial --delay 1 --out /work/.../spatial_d1.json
"""
import argparse
import io
import json
import math
import multiprocessing as mp
import os
import pathlib
import queue
import sys
import time

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

# Stall-detection timeout: a worker that produces NO message (heartbeat, ep,
# or done) for this long is presumed dead/stuck and gets killed+respawned.
# This must NOT be sized off a single episode's total duration: libero_10
# has 520-step episodes and, at observed server round-trip latency
# (~19-20s per k=5 chunk => ~104 chunks/episode), a single episode can
# legitimately take 30+ minutes. Killing on episode-completion alone (no
# heartbeat) would keep restarting a perfectly healthy worker before it
# ever finishes ep 0 -- observed in practice: 3/3 libero_10 shards hit
# "WORKER DIED ... respawning" repeatedly while the policy server log
# showed continuous, healthy /predict traffic the whole time. Heartbeats
# every HEARTBEAT_EVERY_N_STEPS steps keep this timeout meaningful as a
# true liveness check regardless of episode length.
EPISODE_TIMEOUT_S = 15 * 60
HEARTBEAT_EVERY_N_STEPS = 20
POLL_S = 5
# Respawns are shared across the whole shard (all tasks), not per task:
# effectively unlimited but bounded so a truly wedged shard still exits.
MAX_RESPAWNS = 200


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


def run_episode(
    env,
    init_state,
    task_str: str,
    delay: int,
    max_steps: int,
    port: int,
    heartbeat=None,
    stale_state: bool = False,
    log_actions: bool = False,
    log_actions_n: int = 15,
) -> bool:
    """Run one episode. If given, heartbeat(step) is called periodically so a
    caller polling from another process can tell "slow but alive" from
    "actually dead" without waiting for the whole (potentially long) episode
    to finish -- libero_10 episodes take ~500+ steps and, at observed
    per-chunk server latency, can take on the order of 30+ minutes.

    If log_actions is set, prints (flush=True) the first log_actions_n
    full 7-dim action vectors actually sent to env.step -- debug aid for
    diagnosing whether served actions are frozen/near-zero, wrongly scaled,
    or sign-flipped relative to what the env expects.
    """
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    policy = RemotePolicy(port, task_str)
    executor = DelayedChunkExecutor(policy, k=K, delay=delay, stale_state=stale_state)

    for step in range(max_steps):
        arrays = libero_obs_to_arrays(obs)
        images = {"image": arrays["image"], "wrist_image": arrays["wrist_image"]}
        action = executor.act(images, arrays["state"], task_str)
        if log_actions and step < log_actions_n:
            print(f"[log-actions] step {step}: action={action.tolist()}", flush=True)
        obs, _, done, _ = env.step(action.tolist())
        if heartbeat is not None and step % HEARTBEAT_EVERY_N_STEPS == 0:
            heartbeat(step)
        if done:
            return True
    return False


# --- Resume bookkeeping (pure logic, no LIBERO import needed -> unit-testable) ---


def task_done(task_result, num_trials: int) -> bool:
    """True iff a persisted per-task result already covers all trials."""
    return bool(task_result) and task_result.get("trials", 0) >= num_trials


def resume_start_ep(task_result) -> int:
    """Index of the next episode to run, given a task's persisted result."""
    if not task_result:
        return 0
    return len(task_result.get("ep_results", []))


def record_episode(task_result: dict, ep: int, ok: bool) -> dict:
    """Append one in-order episode outcome and recompute successes/trials.

    Episodes are always reported in ascending order starting from the
    worker's start_ep (a fresh or respawned worker only ever runs forward),
    so an out-of-order report indicates a logic bug upstream.
    """
    ep_results = task_result.setdefault("ep_results", [])
    if ep != len(ep_results):
        raise ValueError(
            f"out-of-order episode report: expected ep {len(ep_results)}, got {ep}"
        )
    ep_results.append(bool(ok))
    task_result["successes"] = sum(ep_results)
    task_result["trials"] = len(ep_results)
    return task_result


def _load_prev_results(out_path: pathlib.Path) -> dict:
    if not out_path.exists():
        return {}
    try:
        prev = json.loads(out_path.read_text())
        return prev.get("per_task", {})
    except (json.JSONDecodeError, OSError):
        return {}


def _write_results(out_path: pathlib.Path, suite_name: str, delay: int, results: dict) -> dict:
    """Recompute success_rate from per_task and atomically rewrite --out.

    Schema is unchanged from the non-resumable client:
    {"suite", "delay", "per_task", "success_rate"}, valid on every write
    including partial ones. per_task[name] keeps "successes"/"trials"
    (what Task 9 reads) plus an additive "ep_results" list.
    """
    total_s = sum(r["successes"] for r in results.values())
    total_t = sum(r["trials"] for r in results.values())
    out = {
        "suite": suite_name,
        "delay": delay,
        "per_task": results,
        "success_rate": (total_s / total_t) if total_t else 0.0,
    }
    tmp_path = pathlib.Path(str(out_path) + ".tmp")
    tmp_path.write_text(json.dumps(out, indent=2))
    os.replace(tmp_path, out_path)
    return out


def _run_task_worker(
    bddl_path,
    task_str,
    init_states,
    start_ep,
    num_trials,
    delay,
    max_steps,
    port,
    result_queue,
    stale_state=False,
    log_actions=False,
):
    """Runs episodes [start_ep, num_trials) for one LIBERO task.

    Executed in a spawned child process: imports LIBERO itself (spawn gives
    a clean interpreter/EGL/MuJoCo state per attempt) and creates the env
    once for the whole run of episodes. Puts ("heartbeat", ep_idx, step)
    on result_queue periodically while an episode is in progress (so the
    parent can tell a slow-but-alive episode from a dead one without
    waiting for the whole episode), ("ep", ep_idx, success_bool) after
    each completed episode, then ("done",) once the range is exhausted.
    If the process is killed (native abort) mid-episode, it simply stops
    producing messages; the parent detects that via process-liveness /
    queue-timeout and respawns from the next episode.
    """
    from libero.libero.envs import OffScreenRenderEnv

    import torch

    _orig_torch_load = torch.load

    def _load_full(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _load_full

    # Benchmark-SAFE OffScreenRenderEnv kwargs: hard_reset=True rebuilds the
    # sim on every reset; explicit camera_names and render_gpu_device_id pin
    # the offscreen rendering pipeline.
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview", "robot0_eye_in_hand"],
        control_freq=20,
        hard_reset=True,
        render_gpu_device_id=0,
    )
    env.seed(0)
    try:
        for ep in range(start_ep, num_trials):
            def _heartbeat(step, _ep=ep):
                result_queue.put(("heartbeat", _ep, step))

            ok = run_episode(
                env,
                init_states[ep],
                task_str,
                delay,
                max_steps,
                port,
                heartbeat=_heartbeat,
                stale_state=stale_state,
                log_actions=log_actions,
            )
            result_queue.put(("ep", ep, bool(ok)))
        result_queue.put(("done",))
    finally:
        env.close()


def main():
    from libero.libero import benchmark as libero_benchmark
    from libero.libero import get_libero_path

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
    parser.add_argument(
        "--stale-state",
        action="store_true",
        help=(
            "Naive async: chunk-switch predict call conditions on the same "
            "stale snapshot state as the stale images (fully stale obs), "
            "instead of the true state at the chunk-start step."
        ),
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Run only this task index (0-based) instead of the whole suite. Debug aid.",
    )
    parser.add_argument(
        "--log-actions",
        action="store_true",
        help=(
            "Print (flush=True) the first 15 full action vectors sent to "
            "env.step per episode. Debug aid for diagnosing served-action "
            "issues (frozen/scale/sign) against the live env."
        ),
    )
    args = parser.parse_args()

    # Parent process only needs LIBERO for suite/task metadata (task list,
    # init states, bddl paths) -- all picklable args handed to the worker.
    # It never touches OffScreenRenderEnv/EGL itself.
    suite = libero_benchmark.get_benchmark_dict()[args.suite]()
    max_steps = SUITES[args.suite]

    out_path = pathlib.Path(args.out)
    results = _load_prev_results(out_path)

    ctx = mp.get_context("spawn")
    shard_respawns = 0

    task_ids = [args.task_id] if args.task_id is not None else range(suite.n_tasks)
    for task_id in task_ids:
        task = suite.get_task(task_id)
        task_str = task.language
        task_result = results.get(task.name)

        if task_done(task_result, args.num_trials):
            print(
                f"[{args.suite} d={args.delay}] task {task_id} ({task_str[:40]}) "
                "SKIP (already done, resuming)",
                flush=True,
            )
            continue
        if task_result is None:
            task_result = {"successes": 0, "trials": 0, "ep_results": []}
            results[task.name] = task_result

        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        start_ep = resume_start_ep(task_result)
        while start_ep < args.num_trials:
            if shard_respawns > MAX_RESPAWNS:
                raise RuntimeError(
                    f"Exceeded {MAX_RESPAWNS} worker respawns for shard "
                    f"{args.suite} d={args.delay}; aborting at task {task_id} ep {start_ep}"
                )

            result_queue = ctx.Queue()
            proc = ctx.Process(
                target=_run_task_worker,
                args=(
                    str(bddl),
                    task_str,
                    init_states,
                    start_ep,
                    args.num_trials,
                    args.delay,
                    max_steps,
                    args.port,
                    result_queue,
                    args.stale_state,
                    args.log_actions,
                ),
            )
            proc.start()

            finished = False
            last_progress = time.monotonic()
            while True:
                try:
                    msg = result_queue.get(timeout=POLL_S)
                except queue.Empty:
                    if not proc.is_alive():
                        break  # worker exited (crash or otherwise) with nothing more to read
                    if time.monotonic() - last_progress > EPISODE_TIMEOUT_S:
                        break  # alive but stalled past the per-episode budget
                    continue
                last_progress = time.monotonic()
                if msg[0] == "ep":
                    _, ep_idx, ok = msg
                    record_episode(task_result, ep_idx, ok)
                    _write_results(out_path, args.suite, args.delay, results)
                    start_ep = ep_idx + 1
                    print(
                        f"[{args.suite} d={args.delay}] task {task_id} ({task_str[:40]}) "
                        f"ep {ep_idx}: {'OK' if ok else 'fail'} "
                        f"({task_result['successes']}/{task_result['trials']})",
                        flush=True,
                    )
                elif msg[0] == "done":
                    finished = True
                    break
                elif msg[0] == "heartbeat":
                    _, hb_ep, hb_step = msg
                    print(
                        f"[{args.suite} d={args.delay}] task {task_id} ({task_str[:40]}) "
                        f"heartbeat: ep {hb_ep} step {hb_step}",
                        flush=True,
                    )

            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=60)
            if proc.is_alive():
                # Stuck holding an EGL/GPU context past a SIGTERM -- force it.
                proc.kill()
                proc.join(timeout=60)

            if finished:
                break

            shard_respawns += 1
            print(
                f"WORKER DIED at ep {start_ep} (exitcode={proc.exitcode}), "
                f"respawning (shard respawn {shard_respawns}/{MAX_RESPAWNS})",
                flush=True,
            )

    out = _write_results(out_path, args.suite, args.delay, results)
    print("WROTE", args.out, "success_rate", out["success_rate"], flush=True)


if __name__ == "__main__":
    main()
