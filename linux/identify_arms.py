from damiao_motor import DaMiaoController
import time

with open('/home/oalami/Desktop/openarms/active_cans.txt') as f:
    channels = [line.strip() for line in f if line.strip()]

mapping = {}

for ch in channels:
    input(f'Press Enter to move {ch}...')
    try:
        ctrl = DaMiaoController(channel=ch, bustype='socketcan')
        motors = {}
        for i in range(1, 9):
            motors[i] = ctrl.add_motor(motor_id=i, feedback_id=0x10+i, motor_type='4310')
        for m in motors.values():
            m.enable()
            time.sleep(0.15)
        time.sleep(0.3)
        motors[1].send_cmd_mit(0.3, 0.0, 8.0, 1.0, 0.0)
        time.sleep(2.0)
        motors[1].send_cmd_mit(0.0, 0.0, 8.0, 1.0, 0.0)
        time.sleep(2.0)
        ctrl.disable_all()
    except Exception as e:
        print(f'ERR: {e}')
    
    role = input(f'What is {ch}? (ll=leader_left, lr=leader_right, fl=follower_left, fr=follower_right): ').strip()
    mapping[role] = ch

# Write mapping
with open('/home/oalami/Desktop/openarms/arm_mapping.txt', 'w') as f:
    for role, ch in mapping.items():
        f.write(f'{role}={ch}\n')

print('Mapping saved:')
for role, ch in mapping.items():
    print(f'  {role} -> {ch}')
