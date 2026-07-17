#!/usr/bin/env python3
"""
Replay a single episode's recorded RIGHT ARM (follower) motion directly from
collect.py's raw dataset format (parquet). Only the arm you actually moved
during recording (fr) is replayed -- the fl half is ignored since it was
never actively teleoperated.

Usage:
    python3 replay.py --root ./dataset_quality --episode 3
    python3 replay.py --root ./dataset_100 --episode 7 --dry-run
    python3 replay.py --root ./dataset_quality --episode 3 --source state
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from damiao_motor import DaMiaoController

MOTOR_IDS = range(1, 9)
CONTROL_FREQ = 50
DT = 1.0 / CONTROL_FREQ
ARM_MAPPING_FILE = "/home/oalami/Desktop/openarms/arm_mapping.txt"


def load_mapping():
    mapping = {}
    with open(ARM_MAPPING_FILE) as f:
        for line in f:
            if "=" in line:
                role, ch = line.strip().split("=")
                mapping[role] = ch
    return mapping


def parquet_path(root: Path, idx: int) -> Path:
    return root / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"


def load_episode(root: Path, idx: int) -> pd.DataFrame:
    path = parquet_path(root, idx)
    if not path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {path}")
    df = pd.read_parquet(path)
    df = df.sort_values("frame_index").reset_index(drop=True)
    return df


def setup_follower_motors(channel: str):
    ctrl = DaMiaoController(channel=channel, bustype="socketcan")
    arm_motors = {}
    for i in MOTOR_IDS:
        arm_motors[i] = ctrl.add_motor(motor_id=i, feedback_id=0x10 + i, motor_type="4310")
    for m in arm_motors.values():
        m.enable()
        time.sleep(0.05)
    return ctrl, arm_motors


def extract_fr_positions(row, source: str) -> np.ndarray:
    """
    collect.py layout:
      action            = fl_actions (8) + fr_actions (8)          -- commanded targets
      observation.state = fl_sv (24: pos,vel,torq x8) + fr_sv (24) -- passively-read actual state

    Since only the right arm ('lr' leader -> 'fr' follower) was actively
    moved, we always pull the SECOND half of whichever field is requested.
    """
    if source == "action":
        action = np.asarray(row["action"], dtype=np.float32)
        return action[8:16]  # fr_actions, positions only
    elif source == "state":
        state = np.asarray(row["observation.state"], dtype=np.float32)
        fr_sv = state[24:48]  # fr's [pos, vel, torq] x8
        return fr_sv[0::3]     # pos slots only
    else:
        raise ValueError(f"Unknown source '{source}'")


def replay(root: Path, episode_index: int, dry_run: bool, source: str):
    df = load_episode(root, episode_index)
    print(f"Loaded episode {episode_index} from {root}: {len(df)} rows")
    print(f"Task: {df['task'].iloc[0] if 'task' in df.columns else 'unknown'}")
    print(f"Replay source: {source} (right arm / 'fr' follower only)")

    if dry_run:
        print("Dry run -- printing first 3 fr position vectors, no motor commands sent.")
        for i in range(min(3, len(df))):
            print(f"frame {i}: fr_positions={extract_fr_positions(df.iloc[i], source)}")
        return

    mapping = load_mapping()
    if "fr" not in mapping:
        raise RuntimeError(f"Role 'fr' not found in arm_mapping.txt. Available: {list(mapping.keys())}")
    ctrl, arm_motors = setup_follower_motors(mapping["fr"])

    print("Replaying right arm (fr) in 3 seconds... Ctrl+C to abort.")
    time.sleep(3.0)

    try:
        for _, row in df.iterrows():
            t0 = time.time()
            positions = extract_fr_positions(row, source)

            for motor_idx, i in enumerate(MOTOR_IDS):
                target_pos = float(positions[motor_idx])
                arm_motors[i].send_cmd_mit(target_pos, 0.0, 10.0, 1.0, 0.0)

            elapsed = time.time() - t0
            time.sleep(max(0, DT - elapsed))
    except KeyboardInterrupt:
        print("Replay aborted by user.")
    finally:
        ctrl.disable_all()
        print("Motors disabled.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                    help="Path to raw dataset folder, e.g. ./dataset_quality, ./dataset_100")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--dry-run", action="store_true", help="Print positions instead of moving the arm")
    p.add_argument("--source", type=str, default="action", choices=["action", "state"],
                    help="Replay commanded action targets (default) or passively-read actual state")
    args = p.parse_args()
    replay(args.root, args.episode, args.dry_run, args.source)