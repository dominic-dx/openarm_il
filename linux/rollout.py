#!/usr/bin/env python3
"""
Closed-loop ACT rollout on the physical right-follower (fr) arm.
Mirrors camera/motor setup from collect.py and replay.py, which are
already validated on this hardware.


Usage (dry run, no motors/camera, just checks model loads):
    python3 rollout.py --checkpoint ./checkpoints/run1_first50 --dry-run


Usage (live, single inference query then loop through predicted chunk):
    python3 rollout.py --checkpoint ./checkpoints/run1_first50 --n-action-steps 15


Usage (live, rise to a fixed home position first, then start inference):
    python3 rollout.py --checkpoint ./checkpoints/run1_first50 --n-action-steps 15 \\
        --home-position 0,0,0,0,0,0,0,0
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
WRIST_CAM_SERIAL = "352122273783"

# Deterministic rest pose, one target per motor (radians/whatever unit send_cmd_mit expects).
# Override at runtime with --home-position if this default isn't right for your setup.
DEFAULT_HOME_POSITION = np.zeros(8, dtype=np.float32)


def setup_camera(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, IMAGE_SIZE[1], IMAGE_SIZE[0], rs.format.bgr8, 30)
    pipeline.start(config)
    time.sleep(1.0)
    return pipeline



def get_camera_image(pipeline) -> np.ndarray:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
    image = np.asanyarray(color_frame.get_data())
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mapping():
    mapping = {}
    with open(ARM_MAPPING_FILE) as f:
        for line in f:
            if "=" in line:
                role, ch = line.strip().split("=")
                mapping[role] = ch
    return mapping


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
                      chunk_size=config["chunk_size"], pretrained_backbone=False, use_wrist_cam=config.get("use_wrist_cam", False)).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))
    model.eval()
    return model, config



def preprocess_image(image: np.ndarray, device) -> torch.Tensor:
    resized = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return tensor


def interpolate_positions(start, end, alpha):
    return start + (end - start) * alpha


def ramp_to_target(arm_motors, start_positions, target_positions, ramp_steps, dry_run, label="ramp"):
    """Blocking linear interpolation from start_positions to target_positions over ramp_steps
    control ticks. Used both for the deterministic home rise and the first-action smoothing."""
    for r in range(1, ramp_steps + 1):
        t0 = time.time()
        alpha = r / ramp_steps
        ramped = interpolate_positions(start_positions, target_positions, alpha)

        if dry_run:
            print(f"{label} {r}/{ramp_steps}: positions={ramped}")
        else:
            send_positions(arm_motors, ramped)

        elapsed = time.time() - t0
        time.sleep(max(0, DT - elapsed))
    return target_positions


def run(checkpoint_dir: Path, dry_run: bool, n_action_steps: int, ramp_steps: int,
        home_position: np.ndarray, home_ramp_steps: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, config = load_model(checkpoint_dir, device)
    use_wrist_cam = config.get("use_wrist_cam", False)
    state_mean = torch.tensor(config["norm_stats"]["state_mean"], device=device)
    state_std = torch.tensor(config["norm_stats"]["state_std"], device=device)


    chest_pipeline = None
    wrist_pipeline = None
    ctrl = None
    arm_motors = None


    if not dry_run:
        mapping = load_mapping()
        print("Setting up chest camera...")
        chest_pipeline = setup_camera(CHEST_CAM_SERIAL)
        if use_wrist_cam:
            print("Setting up wrist camera...")
            wrist_pipeline = setup_camera(WRIST_CAM_SERIAL)
        print("Setting up follower motors...")
        ctrl, arm_motors = setup_follower_motors(mapping["fr"])
        print("Setup complete. Starting rollout in 3 seconds... Ctrl+C to abort.")
        time.sleep(3.0)
    else:
        print("DRY RUN: using zeroed dummy image/state, no camera/motor access.")


    # ---- Deterministic rise to a fixed home position, before any inference happens ----
    if home_position is not None and home_ramp_steps > 0:
        print(f"Rising to home position {home_position.tolist()} over {home_ramp_steps} steps...")
        if dry_run:
            current_positions = np.zeros(8, dtype=np.float32)
        else:
            current_state = get_current_state(arm_motors)
            current_positions = current_state[0::3]
        ramp_to_target(arm_motors, current_positions, home_position, home_ramp_steps,
                        dry_run, label="home-ramp")
        print("Reached home position.")


    first_action = True


    try:
        while True:
            if dry_run:
                image = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
                wrist_image = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
                state = np.zeros(24, dtype=np.float32)
                if home_position is not None:
                    state[0::3] = home_position
            else:
                image = get_camera_image(chest_pipeline)
                state = get_current_state(arm_motors)
                if use_wrist_cam:
                    wrist_image = get_camera_image(wrist_pipeline)
                cv2.imwrite(f"/tmp/frame_{int(time.time())}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


            image_t = preprocess_image(image, device)
            state_t = ((torch.from_numpy(state).to(device) - state_mean) / state_std).unsqueeze(0)


            with torch.no_grad():
                t_start = time.time()
                if use_wrist_cam:
                    wrist_t = preprocess_image(wrist_image, device)
                    pred_actions, _, _ = model(image_t, state_t, wrist_image=wrist_t)
                else:
                    pred_actions, _, _ = model(image_t, state_t)
                print(f"Inference took {time.time() - t_start:.2f}s")


            action_chunk = pred_actions.squeeze(0).cpu().numpy()  # (chunk_size, 24)

            if first_action and ramp_steps > 0:
                print(f"Ramping into first action over {ramp_steps} steps...")
                start_positions = state[0::3]  # current pos slots (home position if set above)
                target_positions = action_chunk[0, 0::3]
                ramp_to_target(arm_motors, start_positions, target_positions, ramp_steps,
                                dry_run, label="step-ramp")
                first_action = False


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
    finally:
        if not dry_run and ctrl is not None:
            print("Disabling motors...")
            ctrl.disable_all()
        if chest_pipeline is not None:
            chest_pipeline.stop()
        if wrist_pipeline is not None:
            wrist_pipeline.stop()



def parse_home_position(s: str) -> np.ndarray:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 8:
        raise argparse.ArgumentTypeError(f"--home-position must have exactly 8 comma-separated values, got {len(parts)}")
    return np.array(parts, dtype=np.float32)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n-action-steps", type=int, default=15,
                    help="How many steps of the predicted chunk to execute before re-querying the model")
    p.add_argument("--ramp-steps", type=int, default=20,
                    help="Number of interpolation steps used to smoothly reach the first predicted "
                         "position before normal action execution begins. Set to 0 to disable ramping.")
    p.add_argument("--home-position", type=parse_home_position, default=None,
                    help="Comma-separated 8 values, e.g. '0,0,0,0,0,0,0,0'. If set, the arm rises "
                         "deterministically to this position before any inference/rollout begins. "
                         "Defaults to no home rise if omitted.")
    p.add_argument("--home-ramp-steps", type=int, default=60,
                    help="Number of interpolation steps used for the deterministic rise to "
                         "--home-position. Ignored if --home-position is not set.")
    args = p.parse_args()
    run(args.checkpoint, args.dry_run, args.n_action_steps, args.ramp_steps,
        args.home_position, args.home_ramp_steps)