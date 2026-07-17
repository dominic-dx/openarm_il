from damiao_motor import DaMiaoController

#with open('/root/Desktop/openarms/active_cans.txt') as f:
with open('/home/oalami/Desktop/openarms/active_cans.txt') as f:
    channels = [line.strip() for line in f if line.strip()]

for ch in channels:
    try:
        ctrl = DaMiaoController(channel=ch, bustype='socketcan')
        for i in range(1, 9):
            ctrl.add_motor(motor_id=i, feedback_id=0x10+i, motor_type='4310')
        ctrl.disable_all()
        print(f'{ch} disabled')
    except Exception as e:
        print(f'{ch} ERR: {e}')
