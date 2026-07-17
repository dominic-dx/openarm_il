from damiao_motor import DaMiaoController
import time

mapping = {}
with open('/home/oalami/Desktop/openarms/arm_mapping.txt') as f:
    for line in f:
        if '=' in line:
            role, ch = line.strip().split('=')
            mapping[role] = ch

print("Arm mapping loaded:")
for role, ch in mapping.items():
    print(f"  {role} -> {ch}")

MOTOR_IDS = range(1, 9)
FREQ = 100
DT = 1.0 / FREQ
SCALE = 1.5

LEADER_R_GRIPPER_OPEN  = 0.045
LEADER_R_GRIPPER_CLOSE = 0.850
FOLLOWER_R_GRIPPER_OPEN  = 0.022
FOLLOWER_R_GRIPPER_CLOSE = 1.8

def setup_arm(channel):
    ctrl = DaMiaoController(channel=channel, bustype='socketcan')
    motors = {}
    for i in MOTOR_IDS:
        motors[i] = ctrl.add_motor(motor_id=i, feedback_id=0x10+i, motor_type='4310')
    for m in motors.values():
        m.enable()
        time.sleep(0.3)
    time.sleep(1.0)
    return ctrl, motors

def get_positions(motors):
    pos = {}
    for i in MOTOR_IDS:
        motors[i].send_cmd_mit(0.0, 0.0, 0.0, 0.0, 0.0)
        pos[i] = motors[i].get_states().get('pos', 0.0)
    return pos

print("Setting up arms...")
ll_ctrl, ll = setup_arm(mapping['ll'])
lr_ctrl, lr = setup_arm(mapping['lr'])
fl_ctrl, fl = setup_arm(mapping['fl'])
fr_ctrl, fr = setup_arm(mapping['fr'])

print("Recording initial positions...")
time.sleep(0.5)
ll_init = get_positions(ll)
lr_init = get_positions(lr)
fl_init = get_positions(fl)
fr_init = get_positions(fr)

print(f"Starting teleoperation at {FREQ}Hz. Ctrl+C to stop.")
try:
    while True:
        t0 = time.time()
        left_pos  = get_positions(ll)
        right_pos = get_positions(lr)

        for i in MOTOR_IDS:
            if i == 8:
                # Left gripper: delta-based, inverted
                left_delta = (left_pos[i] - ll_init[i]) * -1
                fl_t = fl_init[i] + left_delta

                # Right gripper: absolute position mapping
                leader_raw = right_pos[i]
                leader_frac = (leader_raw - LEADER_R_GRIPPER_OPEN) / (LEADER_R_GRIPPER_CLOSE - LEADER_R_GRIPPER_OPEN)
                leader_frac = max(0.0, min(1.0, leader_frac))
                fr_t = FOLLOWER_R_GRIPPER_OPEN + leader_frac * (FOLLOWER_R_GRIPPER_CLOSE - FOLLOWER_R_GRIPPER_OPEN)
            else:
                left_delta  = (left_pos[i]  - ll_init[i]) * SCALE
                right_delta = (right_pos[i] - lr_init[i]) * SCALE
                fl_t = fl_init[i] + left_delta
                fr_t = fr_init[i] + right_delta

            fl[i].send_cmd_mit(fl_t, 0.0, 10.0, 1.0, 0.0)
            fr[i].send_cmd_mit(fr_t, 0.0, 20.0, 2.0, 0.0)

        elapsed = time.time() - t0
        time.sleep(max(0, DT - elapsed))

except KeyboardInterrupt:
    print("\nDisabling all arms...")
    for ctrl in [ll_ctrl, lr_ctrl, fl_ctrl, fr_ctrl]:
        ctrl.disable_all()
    print("Done.")
