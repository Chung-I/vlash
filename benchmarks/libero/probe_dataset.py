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
