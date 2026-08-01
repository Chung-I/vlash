#!/usr/bin/env python
"""Decisive offline probe for the pi05_libero 0%-success investigation.

serve_policy.inject_dataset_stats is now wired to populate the normalization
buffers of `lerobot/pi05_libero` (whose model.safetensors ships none) from
HuggingFaceVLA/libero's dataset stats. This script checks whether, once
that injection happens, the served policy actually produces sane action
predictions -- or whether some other defect (weight-loading gap, image
preprocessing mismatch, tokenizer difference, ...) still poisons inference.

For each of a few dataset frames (mid-episode, spread across episodes) we:
  1. Build the policy input batch exactly like serve_policy.py's /predict
     route does (images float CHW in [0, 1], state float32, task string).
  2. Run predict_action_chunk and compare the first 5 predicted actions
     against the actual next-5 recorded actions from the dataset.
  3. Repeat the same comparison for our own fine-tuned checkpoint (which
     carries its own baked-in, non-inf stats -- no injection needed) as a
     working reference.

If the fine-tuned model tracks ground truth but pi05_libero does not, the
defect is still on the pi05_libero serving path (weight remap / image prep /
tokenizer). If neither tracks ground truth, suspect the eval harness itself.
If both track it, serving is fine and the recorded 0% success has an
environment/protocol-side cause.

Run in the vlash env on nano4 (dev partition, 1 GPU):
    python benchmarks/libero/debug_predict_probe.py
"""
import torch
import torch.nn as nn

from benchmarks.libero.libero_config import DATASET_REPO_ID
from benchmarks.libero.serve_policy import inject_dataset_stats, load_policy

FINETUNED_CKPT = (
    "/work/roboleon1295/vlash/outputs/train/async_libero/checkpoints/030000/pretrained_model"
)


def load_policy_capturing_keys(checkpoint: str):
    """Like load_policy(), but also captures the missing/unexpected key
    lists from the top-level `instance.load_state_dict(...)` call inside
    PI05Policy.from_pretrained, by temporarily monkeypatching
    nn.Module.load_state_dict. from_pretrained already raises on
    unexpected keys, so if we get a policy back there were none -- but
    missing_keys are otherwise silently swallowed, and we want that
    evidence without permanently changing modeling_pi05.py."""
    captured = {}
    orig_load_state_dict = nn.Module.load_state_dict

    def _capturing(self, state_dict, strict=True, **kwargs):
        result = orig_load_state_dict(self, state_dict, strict=strict, **kwargs)
        captured["missing_keys"] = list(getattr(result, "missing_keys", []) or [])
        captured["unexpected_keys"] = list(getattr(result, "unexpected_keys", []) or [])
        return result

    nn.Module.load_state_dict = _capturing
    try:
        policy = load_policy(checkpoint)
    finally:
        nn.Module.load_state_dict = orig_load_state_dict
    return policy, captured


def build_batch(frame: dict, camera_keys: list[str], device) -> dict:
    """Build a policy input batch from one LeRobotDataset frame, matching
    serve_policy.py's /predict route: images as float CHW in [0, 1] with a
    batch dim, state float32 with a batch dim, task as a length-1 list of
    strings. (LeRobotDataset already decodes video frames as float CHW in
    [0, 1]; the server instead converts uint8 HWC -> float CHW / 255 -- both
    land in the same representation.)"""
    batch = {}
    for key in camera_keys:
        img = frame[key]
        if img.dtype != torch.float32:
            img = img.float()
        batch[key] = img.unsqueeze(0).to(device)
    batch["observation.state"] = frame["observation.state"].unsqueeze(0).to(device).float()
    batch["task"] = [str(frame["task"])]
    return batch


def pick_sample_idxs(dataset, n_samples: int = 3) -> list[int]:
    """Pick n_samples frame indices, each mid-episode (>=5 steps of runway
    before the episode ends) from spread-out episodes."""
    total_eps = dataset.meta.total_episodes
    step = max(1, total_eps // 20)
    idxs = []
    for ep_idx in range(0, total_eps, step):
        ep = dataset.meta.episodes[ep_idx]
        start, end = ep["dataset_from_index"], ep["dataset_to_index"]
        if end - start > 20:
            idxs.append(start + (end - start) // 2)
        if len(idxs) >= n_samples:
            break
    return idxs


def run_probe(policy, dataset, camera_keys, sample_idxs, label: str):
    device = next(policy.parameters()).device
    print(f"\n=== {label} ===", flush=True)
    for idx in sample_idxs:
        frame = dataset[idx]
        ep_idx = frame["episode_index"].item()
        batch = build_batch(frame, camera_keys, device)
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)
        pred = chunk[0, :5].float().cpu().numpy()

        gt = []
        for j in range(5):
            nxt = dataset[idx + j]
            if nxt["episode_index"].item() != ep_idx:
                break
            gt.append(nxt["action"].numpy())
        import numpy as np

        gt = np.stack(gt) if gt else np.zeros((0, pred.shape[-1]))
        n = min(len(gt), len(pred))
        mae = float(np.mean(np.abs(pred[:n] - gt[:n]))) if n > 0 else float("nan")

        print(f"idx={idx} ep={ep_idx} task={frame['task']!r}", flush=True)
        print(f"  pred[:5]=\n{pred}", flush=True)
        print(f"  gt[:5]=\n{gt}", flush=True)
        print(f"  MAE={mae:.4f}", flush=True)


def print_buffer_stats(policy, label: str):
    state_buf = policy.normalize_inputs.buffer_observation_state
    action_norm_buf = policy.normalize_targets.buffer_action
    action_unnorm_buf = policy.unnormalize_outputs.buffer_action
    print(f"[{label}] state mean[:3]={state_buf['mean'].flatten()[:3].tolist()}", flush=True)
    print(f"[{label}] state std[:3]={state_buf['std'].flatten()[:3].tolist()}", flush=True)
    print(
        f"[{label}] action(normalize_targets) mean[:3]={action_norm_buf['mean'].flatten()[:3].tolist()}",
        flush=True,
    )
    print(
        f"[{label}] action(normalize_targets) std[:3]={action_norm_buf['std'].flatten()[:3].tolist()}",
        flush=True,
    )
    print(
        f"[{label}] action(unnormalize_outputs) mean[:3]={action_unnorm_buf['mean'].flatten()[:3].tolist()}",
        flush=True,
    )
    print(
        f"[{label}] action(unnormalize_outputs) std[:3]={action_unnorm_buf['std'].flatten()[:3].tolist()}",
        flush=True,
    )


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading dataset frames from {DATASET_REPO_ID}", flush=True)
    dataset = LeRobotDataset(DATASET_REPO_ID)
    camera_keys = dataset.meta.camera_keys
    print(f"camera_keys={camera_keys}", flush=True)

    sample_idxs = pick_sample_idxs(dataset)
    print(f"sample_idxs={sample_idxs}", flush=True)

    print("\n########## RELEASED lerobot/pi05_libero ##########", flush=True)
    policy, captured = load_policy_capturing_keys("lerobot/pi05_libero")
    print(f"weight-load missing_keys ({len(captured['missing_keys'])}): {captured['missing_keys']}", flush=True)
    print(
        f"weight-load unexpected_keys ({len(captured['unexpected_keys'])}): {captured['unexpected_keys']}",
        flush=True,
    )

    print_buffer_stats(policy, "pi05_libero BEFORE injection")
    inject_dataset_stats(policy, repo_id=DATASET_REPO_ID)
    print_buffer_stats(policy, "pi05_libero AFTER injection")

    run_probe(policy, dataset, camera_keys, sample_idxs, "pi05_libero predictions vs ground truth")

    print("\n########## OUR FINE-TUNED CHECKPOINT (baked-in stats, no injection) ##########", flush=True)
    ft_policy, ft_captured = load_policy_capturing_keys(FINETUNED_CKPT)
    print(
        f"finetuned weight-load missing_keys ({len(ft_captured['missing_keys'])}): {ft_captured['missing_keys']}",
        flush=True,
    )
    print_buffer_stats(ft_policy, "finetuned checkpoint")
    run_probe(ft_policy, dataset, camera_keys, sample_idxs, "finetuned checkpoint predictions vs ground truth")

    print("\nPROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
