#!/usr/bin/env python3
"""
Closed-loop ACT rollout on the physical right-follower (fr) arm.
Mirrors camera/motor setup from collect.py and replay.py, which are
already validated on this hardware.

Usage (dry run, no motors/camera, just checks model loads):
    python3 rollout.py --checkpoint ./checkpoints/run1_first50 --dry-run

Usage (live, single inference query then loop through predicted chunk):
    python3 rollout.py --checkpoint ./checkpoints/run1_first50 --n-action-steps 15
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from model import ACTModel
from damiao_motor import DaMiaoController

MOTOR_IDS = range(1, 9)
CONTROL_FREQ = 30
DT = 1.0 / CONTROL_FREQ
ARM_MAPPING_FILE = "/home/oalami/Desktop/openarms/arm_mapping.txt"
IMAGE_SIZE = (480, 848)  # (H, W), matches training preprocessing

CHEST_CAM_SERIAL = "254622073959"


def setup_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(CHEST_CAM_SERIAL)
    config.enable_stream(rs.stream.color, IMAGE_SIZE[1], IMAGE_SIZE[0], rs.format.bgr8, 30)
    pipeline.start(config)
    time.sleep(1.0)  # let auto-exposure settle
    return pipeline

def load_mapping():
    mapping = {}
    with open(ARM_MAPPING_FILE) as f:
        for line in f:
            if "=" in line:
                role, ch = line.strip().split("=")
                mapping[role] = ch
    return mapping

def get_chest_image(pipeline) -> np.ndarray:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
    image = np.asanyarray(color_frame.get_data())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def setup_follower_motors(channel: str):
    ctrl = DaMiaoController(channel=channel, bustype="socketcan")
    arm_motors = {}
    for i in MOTOR_IDS:
        arm_motors[i] = ctrl.add_motor(motor_id=i, feedback_id=0x10 + i, motor_type="4310")
    for m in arm_motors.values():
        m.enable()
        time.sleep(0.05)
    return ctrl, arm_motors


def get_current_state(arm_motors) -> np.ndarray:
    """Mirrors the 24-dim [pos, vel, torq] x 8 motors layout used in training."""
    state_vec = []
    for i in MOTOR_IDS:
        s = arm_motors[i].get_states()
        state_vec.extend([s.get("pos", 0.0), s.get("vel", 0.0), s.get("torq", 0.0)])
    return np.array(state_vec, dtype=np.float32)


def send_positions(arm_motors, positions: np.ndarray):
    """positions: 8-length array, one target position per motor."""
    for motor_idx, i in enumerate(MOTOR_IDS):
        arm_motors[i].send_cmd_mit(float(positions[motor_idx]), 0.0, 10.0, 1.0, 0.0)


def load_model(checkpoint_dir: Path, device):
    config = json.loads((checkpoint_dir / "config.json").read_text())
    model = ACTModel(state_dim=config["state_dim"], action_dim=config["action_dim"],
                      chunk_size=config["chunk_size"], pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))
    model.eval()
    return model, config


def preprocess_image(image: np.ndarray, device) -> torch.Tensor:
    resized = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return tensor


def run(checkpoint_dir: Path, dry_run: bool, n_action_steps: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, config = load_model(checkpoint_dir, device)
    state_mean = torch.tensor(config["norm_stats"]["state_mean"], device=device)
    state_std = torch.tensor(config["norm_stats"]["state_std"], device=device)

    pipeline = None
    ctrl = None
    arm_motors = None

    if not dry_run:
        mapping = load_mapping()
        print("Setting up camera...")
        pipeline = setup_camera()
        print("Setting up follower motors...")
        ctrl, arm_motors = setup_follower_motors(mapping["fr"])
        print("Setup complete. Starting rollout in 3 seconds... Ctrl+C to abort.")
        time.sleep(3.0)

    else:
        print("DRY RUN: using zeroed dummy image/state, no camera/motor access.")

    try:
        while True:
            if dry_run:
                image = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
                state = np.zeros(24, dtype=np.float32)
            else:
                image = get_chest_image(pipeline)
                state = get_current_state(arm_motors)
                cv2.imwrite(f"/tmp/frame_{int(time.time())}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            image_t = preprocess_image(image, device)
            state_t = ((torch.from_numpy(state).to(device) - state_mean) / state_std).unsqueeze(0)

            with torch.no_grad():
                pred_actions, _, _ = model(image_t, state_t)
            action_chunk = pred_actions.squeeze(0).cpu().numpy()  # (chunk_size, 24)

            steps_to_run = min(n_action_steps, action_chunk.shape[0])
            for t in range(steps_to_run):
                t0 = time.time()
                positions = action_chunk[t, 0::3]  # pos slots only, matches training packing

                if dry_run:
                    print(f"step {t}: positions={positions}")
                else:
                    send_positions(arm_motors, positions)

                elapsed = time.time() - t0
                time.sleep(max(0, DT - elapsed))

            if dry_run:
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nRollout stopped by user.")
    finally:
        if not dry_run and ctrl is not None:
            print("Disabling motors...")
            ctrl.disable_all()
        if pipeline is not None:
            pipeline.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n-action-steps", type=int, default=15,
                    help="How many steps of the predicted chunk to execute before re-querying the model")
    args = p.parse_args()
    run(args.checkpoint, args.dry_run, args.n_action_steps)