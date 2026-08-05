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
from lerobot.configs.policies import PreTrainedConfig

from benchmarks.libero.libero_config import IMAGE_KEYS, K, SERVER_PORT
from vlash.policies.factory import get_policy_class


def load_policy(checkpoint: str):
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.compile_model = False
    policy_cls = get_policy_class("pi05")
    policy = policy_cls.from_pretrained(pretrained_name_or_path=checkpoint, config=cfg)
    return policy.to(policy.config.device).eval()


def inject_dataset_stats(policy, repo_id: str | None = None, stats: dict | None = None) -> None:
    """Populate normalization buffers from dataset stats when the checkpoint
    ships none (new-pipeline checkpoints keep stats outside the weights;
    e.g. lerobot/pi05_libero's model.safetensors has no normalization
    buffers and its processor JSONs have empty `features`).

    Refuses to overwrite finite (trained-in) buffers, so this is a no-op
    (raises) if pointed at a checkpoint that already has real stats.

    Args:
        policy: A PI05Policy (or compatible) instance exposing
            normalize_inputs / normalize_targets / unnormalize_outputs
            submodules with buffer_observation_state / buffer_action
            ParameterDicts (see vlash/policies/normalize.py).
        repo_id: HF dataset repo id to load stats from via
            `LeRobotDatasetMetadata(repo_id).stats`. Ignored if `stats` is
            given directly.
        stats: Pre-loaded stats dict, keyed by feature name
            (e.g. "observation.state", "action") -> {"mean": ..., "std": ...}.
            Passing this directly (bypassing `repo_id`) avoids network
            access, which is what tests use.
    """
    if stats is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        stats = LeRobotDatasetMetadata(repo_id).stats

    targets = [
        (policy.normalize_inputs.buffer_observation_state, "observation.state"),
        (policy.normalize_targets.buffer_action, "action"),
        (policy.unnormalize_outputs.buffer_action, "action"),
    ]
    for buffer, key in targets:
        mean = buffer["mean"]
        std = buffer["std"]
        if torch.isfinite(mean).all():
            raise RuntimeError(
                f"normalization buffers for {key} are already populated; "
                "refusing to overwrite a trained checkpoint's stats"
            )
        mean.data = torch.as_tensor(stats[key]["mean"], dtype=torch.float32, device=mean.device)
        std.data = torch.as_tensor(stats[key]["std"], dtype=torch.float32, device=std.device)
        # NOTE: plain print(flush=True), not logging.info -- lerobot installs
        # root logging handlers at import time, so a later
        # logging.basicConfig(level=logging.INFO) in main() is a no-op and
        # INFO records get silently dropped. print is the only reliable way
        # to get this evidence line into the job log.
        print(
            f"[inject_dataset_stats] populated buffers for {key}: "
            f"mean[:3]={mean.detach().flatten()[:3].tolist()}",
            flush=True,
        )


def inject_openpi_quantile_stats(policy, norm_stats_json: str) -> None:
    """Inject openpi's QUANTILE normalization as pseudo mean/std.

    The released openpi pi05 checkpoints normalize state/actions with
    quantiles: x_norm = 2*(x - q01)/(q99 - q01) - 1. The lerobot port declares
    MEAN_STD mode, so setting mean := (q01+q99)/2 and std := (q99-q01)/2 makes
    (x - mean)/std reproduce the quantile formula EXACTLY (and the inverse for
    unnormalization). Injecting true dataset mean/std here (the previous fix)
    mis-scales every state/action against quantile-trained weights -- measured
    ~50% SR on libero_spatial vs ~98% with correct scaling.
    """
    import json

    ns = json.load(open(norm_stats_json))["norm_stats"]

    def pseudo(feat):
        q01 = torch.tensor(ns[feat]["q01"], dtype=torch.float32)
        q99 = torch.tensor(ns[feat]["q99"], dtype=torch.float32)
        return (q01 + q99) / 2.0, (q99 - q01) / 2.0

    targets = [
        (policy.normalize_inputs.buffer_observation_state, "state"),
        (policy.normalize_targets.buffer_action, "actions"),
        (policy.unnormalize_outputs.buffer_action, "actions"),
    ]
    for buffer, feat in targets:
        mean, std = pseudo(feat)
        if torch.isfinite(buffer["mean"]).all():
            raise RuntimeError("stats already populated; refusing to overwrite")
        buffer["mean"].data = mean.to(buffer["mean"].device)
        buffer["std"].data = std.to(buffer["std"].device)
        print(f"[inject_openpi_quantile_stats] {feat}: mean[:3]={mean[:3].tolist()} std[:3]={std[:3].tolist()}", flush=True)


def make_app(policy) -> Flask:
    app = Flask("vlash_libero_server")
    device = next(policy.parameters()).device
    # RTC (arXiv 2506.07339) serving support: per-env cache of the previous
    # chunk in MODEL (normalized) space. Keyed by client-provided env_id; the
    # client uses a fresh env_id per episode so the cache is episode-local.
    rtc_prev: dict[int, torch.Tensor] = {}

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
        if "rtc" in data.files and int(data["rtc"]):
            # guided sampling runs a vjp per step -> must NOT be under
            # torch.inference_mode (inference tensors cannot enter autograd)
            env_id = int(data["rtc_env_id"])
            delay = int(data["rtc_delay"])
            executed = int(data["rtc_executed"])
            prev = rtc_prev.get(env_id)
            env_chunk, model_chunk = policy.predict_action_chunk_rtc(
                batch, prev, delay, executed
            )
            rtc_prev[env_id] = model_chunk
            actions = env_chunk.detach().float().cpu().numpy()
        else:
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(batch)
            take = chunk.shape[1] if "full" in data.files else K
            actions = chunk[0, :take].float().cpu().numpy()
        buf = io.BytesIO()
        np.savez(buf, actions=actions)
        return buf.getvalue()

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    parser.add_argument(
        "--stats-openpi-json",
        default=None,
        help=(
            "Path to an openpi norm_stats.json; injects quantile-parity pseudo "
            "mean/std (exactly reproduces openpi quantile normalization). "
            "Takes precedence over --stats-dataset."
        ),
    )
    parser.add_argument(
        "--stats-dataset",
        default=None,
        help=(
            "HF dataset repo id (e.g. HuggingFaceVLA/libero) to pull normalization "
            "stats from when the checkpoint ships none. Default: don't inject."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    policy = load_policy(args.checkpoint)
    if args.stats_openpi_json:
        inject_openpi_quantile_stats(policy, args.stats_openpi_json)
    elif args.stats_dataset:
        inject_dataset_stats(policy, repo_id=args.stats_dataset)
    app = make_app(policy)
    app.run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
