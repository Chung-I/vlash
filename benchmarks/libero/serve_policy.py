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
