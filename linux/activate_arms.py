import can
import time
import subprocess
from damiao_motor import DaMiaoController

import subprocess
result = subprocess.run(['ip', 'link'], capture_output=True, text=True)
CHANNELS = [line.split(':')[1].strip() for line in result.stdout.split('\n') if 'link/can' in result.stdout and ':' in line and 'can' in line.split(':')[1] if len(line.split(':')) > 1]
MOTOR_IDS = range(1, 9)

def bring_up(ch):
    subprocess.run(['sudo', 'ip', 'link', 'set', ch, 'down'], capture_output=True)
    subprocess.run(['sudo', 'ip', 'link', 'set', ch, 'up', 'type', 'can', 'bitrate', '1000000'], capture_output=True)
    time.sleep(0.3)

def test_channel(ch):
    try:
        bus = can.Bus(interface='socketcan', channel=ch, bitrate=1000000)
        msg = can.Message(arbitration_id=0x01, data=[0xFF]*7+[0xFC], is_extended_id=False)
        bus.send(msg)
        reply = bus.recv(timeout=0.5)
        bus.shutdown()
        return reply is not None
    except:
        return False

def activate_arm(ch):
    ctrl = DaMiaoController(channel=ch, bustype='socketcan')
    motors = {}
    for i in MOTOR_IDS:
        motors[i] = ctrl.add_motor(motor_id=i, feedback_id=0x10+i, motor_type='4310')
    for m in motors.values():
        m.enable()
        time.sleep(0.2)
    time.sleep(0.5)
    for m in motors.values():
        m.send_cmd_mit(target_position=0.0, target_velocity=0.0, stiffness=5.0, damping=0.5, feedforward_torque=0.0)
        time.sleep(0.15)
    return ctrl, motors

# Main
print("Bringing up interfaces...")
for ch in CHANNELS:
    bring_up(ch)

print("Detecting active arms...")
active = {}
for ch in CHANNELS:
    if test_channel(ch):
        print(f"  {ch} -> arm detected")
        active[ch] = None
    else:
        print(f"  {ch} -> no response")

print(f"\nActivating {len(active)} arms...")
controllers = {}
for ch in active:
    try:
        ctrl, motors = activate_arm(ch)
        controllers[ch] = (ctrl, motors)
        states = {i: m.get_states() for i, m in motors.items()}
        print(f"  {ch} -> ENABLED, motor states: {states}")
    except Exception as e:
        print(f"  {ch} -> ERR: {e}")

print(f"\nDone. {len(controllers)} arms active.")
print("Press Ctrl+C to disable all.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    for ch, (ctrl, _) in controllers.items():
        ctrl.disable_all()
        print(f"{ch} disabled")
