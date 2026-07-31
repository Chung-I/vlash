"""Micro-benchmark: isolate scene-vs-context for the libero_10 slowdown.

Builds ONE libero task env with the exact (or overridden) kwargs used by
benchmarks/libero/eval_client.py's _run_task_worker, steps it N times with
the dummy action, and prints per-step wall time (flush=True). No policy
server, no subprocess wrapper, no multiprocessing spawn context -- this is a
single plain process, so any slowness must come from the sim/render stack
itself (or from something ambient like MUJOCO_GL/EGL state), not from the
shard's multiprocessing/spawn/policy-server context.

Each invocation does exactly one suite + one kwargs variant so different
MUJOCO_GL values etc. can be exercised via separate process launches within
the same sbatch job (MUJOCO_GL must be set before the process starts).

Debug-only script for the libero_10 perf investigation
(.superpowers/sdd/2026-07-30-libero-success-rate-repro/); not part of the
production eval pipeline.
"""
import argparse
import os
import time
import pathlib

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def build_env(suite_name, task_id, camera_size, control_freq, hard_reset,
              set_render_gpu_id, render_gpu_device_id):
    from libero.libero import benchmark as libero_benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = libero_benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = suite.get_task_init_states(task_id)

    kwargs = dict(
        bddl_file_name=str(bddl),
        camera_heights=camera_size,
        camera_widths=camera_size,
        camera_names=["agentview", "robot0_eye_in_hand"],
        control_freq=control_freq,
        hard_reset=hard_reset,
    )
    if set_render_gpu_id:
        kwargs["render_gpu_device_id"] = render_gpu_device_id
    print(f"[{suite_name}] env kwargs = {kwargs}", flush=True)

    t0 = time.perf_counter()
    env = OffScreenRenderEnv(**kwargs)
    t1 = time.perf_counter()
    print(f"[{suite_name}] OffScreenRenderEnv() ctor: {t1 - t0:.3f}s", flush=True)

    env.seed(0)
    t2 = time.perf_counter()
    env.reset()
    t3 = time.perf_counter()
    obs = env.set_init_state(init_states[0])
    t4 = time.perf_counter()
    print(f"[{suite_name}] reset(): {t3 - t2:.3f}s  set_init_state(): {t4 - t3:.3f}s", flush=True)
    return env, obs, task.language


def step_loop(suite_name, label, env, obs, n):
    times = []
    for step in range(n):
        t0 = time.perf_counter()
        obs, _, done, _ = env.step(DUMMY_ACTION)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"[{suite_name}|{label}] step {step:03d}: {dt*1000:.1f} ms", flush=True)
        if done:
            print(f"[{suite_name}|{label}] episode done early at step {step}", flush=True)
            break
    n_done = len(times)
    avg = sum(times) / n_done if n_done else float("nan")
    print(
        f"[{suite_name}|{label}] SUMMARY n={n_done} avg={avg*1000:.1f}ms "
        f"min={min(times)*1000:.1f}ms max={max(times)*1000:.1f}ms "
        f"total={sum(times):.2f}s",
        flush=True,
    )
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=[
        "libero_spatial", "libero_object", "libero_goal", "libero_10"])
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--camera-size", type=int, default=256)
    ap.add_argument("--control-freq", type=int, default=20)
    ap.add_argument("--hard-reset", dest="hard_reset", action="store_true", default=True)
    ap.add_argument("--no-hard-reset", dest="hard_reset", action="store_false")
    ap.add_argument("--omit-render-gpu-id", action="store_true",
                     help="do not pass render_gpu_device_id at all")
    ap.add_argument("--render-gpu-device-id", type=int, default=0)
    args = ap.parse_args()

    print(
        f"PID {os.getpid()} MUJOCO_GL={os.environ.get('MUJOCO_GL')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"variant={args.label!r} suite={args.suite}",
        flush=True,
    )

    env, obs, task_str = build_env(
        args.suite,
        args.task_id,
        camera_size=args.camera_size,
        control_freq=args.control_freq,
        hard_reset=args.hard_reset,
        set_render_gpu_id=not args.omit_render_gpu_id,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    try:
        step_loop(args.suite, args.label, env, obs, args.n)
    finally:
        env.close()
    print(f"DONE variant={args.label!r} suite={args.suite}", flush=True)


if __name__ == "__main__":
    main()
