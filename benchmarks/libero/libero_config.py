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
