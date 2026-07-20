#!/usr/bin/env python3
"""
Interactive arm position reader.

Puts the arm into a backdrivable (zero-gain) state so you can manually
move it by hand, and continuously prints live joint positions to the
terminal. Use this to find/note down a good home position, then pass
those 8 values into rollout.py's --home-position flag.

Usage:
    python3 read_positions.py --arm fr
    python3 read_positions.py --arm fl --hz 5

Press Ctrl+C to stop and print a final copy-pasteable --home-position string.
"""

import argparse
import time

from damiao_motor import DaMiaoController

MOTOR_IDS = range(1, 9)
ARM_MAPPING_FILE = "/home/oalami/Desktop/openarms/arm_mapping.txt"


def load_mapping():
    mapping = {}
    with open(ARM_MAPPING_FILE) as f:
        for line in f:
            if "=" in line:
                role, ch = line.strip().split("=")
                mapping[role] = ch
    return mapping


def setup_backdrivable_motors(channel: str):
    """Enables motors but sends zero stiffness/damping so the arm can be
    moved freely by hand while feedback packets still stream back."""
    ctrl = DaMiaoController(channel=channel, bustype="socketcan")
    arm_motors = {}
    for i in MOTOR_IDS:
        arm_motors[i] = ctrl.add_motor(motor_id=i, feedback_id=0x10 + i, motor_type="4310")
    for m in arm_motors.values():
        m.enable()
        time.sleep(0.05)
    return ctrl, arm_motors


def read_positions(arm_motors):
    """Sends a zero-gain MIT command to trigger a feedback packet without
    resisting motion, then reads back pos/vel/torq per motor."""
    positions, velocities, torques = [], [], []
    for i in MOTOR_IDS:
        arm_motors[i].send_cmd_mit(0.0, 0.0, 0.0, 0.0, 0.0)
        s = arm_motors[i].get_states()
        positions.append(s.get("pos", 0.0))
        velocities.append(s.get("vel", 0.0))
        torques.append(s.get("torq", 0.0))
    return positions, velocities, torques


def run(arm_role: str, hz: float):
    mapping = load_mapping()
    if arm_role not in mapping:
        raise ValueError(f"Arm role '{arm_role}' not found in {ARM_MAPPING_FILE}. "
                          f"Available roles: {list(mapping.keys())}")

    channel = mapping[arm_role]
    print(f"Connecting to arm '{arm_role}' on channel {channel}...")
    ctrl, arm_motors = setup_backdrivable_motors(channel)
    print("Motors enabled with zero stiffness/damping -- arm is backdrivable.")
    print("Move the arm by hand to the pose you want, watch positions settle, "
          "then Ctrl+C to capture it.\n")

    dt = 1.0 / hz
    last_positions = [0.0] * 8

    try:
        while True:
            t0 = time.time()
            positions, velocities, torques = read_positions(arm_motors)
            last_positions = positions

            pos_str = "  ".join(f"m{i+1}={p:+.4f}" for i, p in enumerate(positions))
            print(f"\r{pos_str}", end="", flush=True)

            elapsed = time.time() - t0
            time.sleep(max(0, dt - elapsed))
    except KeyboardInterrupt:
        print("\n\nStopped.")
        home_str = ",".join(f"{p:.4f}" for p in last_positions)
        print("Last captured positions (one per motor, in MOTOR_IDS order):")
        print(f"  {last_positions}")
        print("\nCopy-paste ready for rollout.py:")
        print(f'  --home-position={home_str}')
    finally:
        print("Disabling motors...")
        ctrl.disable_all()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", type=str, default="fr",
                    help="Arm role as defined in arm_mapping.txt, e.g. 'fr' or 'fl'")
    p.add_argument("--hz", type=float, default=10.0,
                    help="How many times per second to poll and print positions")
    args = p.parse_args()
    run(args.arm, args.hz)