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
        logging.info(
            "[inject_dataset_stats] populated buffers for %s: mean[:3]=%s",
            key,
            mean.detach().flatten()[:3].tolist(),
        )


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
        actions = chunk[0, :K].float().cpu().numpy()
        buf = io.BytesIO()
        np.savez(buf, actions=actions)
        return buf.getvalue()

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
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
    if args.stats_dataset:
        inject_dataset_stats(policy, repo_id=args.stats_dataset)
    app = make_app(policy)
    app.run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
