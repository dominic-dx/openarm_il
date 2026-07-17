
#!/bin/bash
sudo rmmod gs_usb 2>/dev/null
sleep 1
sudo insmod /lib/modules/6.8.0-124-generic/kernel/drivers/net/can/usb/gs_usb.ko
sleep 2

# Bring up all CAN interfaces
for iface in $(ip -o link show type can | awk '{print $2}' | tr -d ':'); do
    sudo ip link set "$iface" down 2>/dev/null
    sudo ip link set "$iface" up type can bitrate 1000000
    echo "$iface up"
done

echo "Waiting for motors to power up..."
sleep 5

echo "Detecting active arms..."
ACTIVE=""
for iface in $(ip -o link show type can | awk '{print $2}' | tr -d ':'); do
    RESPONSE=$(python3 -c "
import sys
sys.path.insert(0, '/home/$USER/Desktop/openarms/aloha-openarm')
try:
    from damiao_motor import DaMiaoController
    ctrl = DaMiaoController(channel='$iface', bustype='socketcan')
    ctrl.enable_all()
    import time; time.sleep(0.3)
    fb = ctrl.get_feedback(1)
    ctrl.disable_all()
    ctrl.bus.shutdown()
    print('ok' if fb else 'none')
except Exception as e:
    print('err')
" 2>/dev/null)
    if [ "$RESPONSE" = "ok" ]; then
        echo "  $iface -> arm detected"
        ACTIVE="$ACTIVE $iface"
    else
        echo "  $iface -> no response"
    fi
done

echo $ACTIVE | tr ' ' '\n' | grep -v '^$' > ~/Desktop/openarms/active_cans.txt
echo "Active channels written to active_cans.txt:"
cat ~/Desktop/openarms/active_cans.txt
#!/bin/bash
sudo rmmod gs_usb 2>/dev/null
sleep 1
sudo insmod /lib/modules/6.8.0-124-generic/kernel/drivers/net/can/usb/gs_usb.ko
sleep 2

for iface in $(ip -o link show type can | awk '{print $2}' | tr -d ':'); do
    sudo ip link set "$iface" down 2>/dev/null
    sudo ip link set "$iface" up type can bitrate 1000000
    echo "$iface up"
done

echo "Waiting for motors to power up..."
sleep 3

echo "Detecting active arms..."
ACTIVE=""
for iface in $(ip -o link show type can | awk '{print $2}' | tr -d ':'); do
    RESPONSE=$(python3 -c "
from damiao_motor import DaMiaoController
import time
try:
    ctrl = DaMiaoController(channel='$iface', bustype='socketcan')
    m = ctrl.add_motor(motor_id=1, feedback_id=0x11, motor_type='4310')
    m.enable()
    time.sleep(0.3)
    m.send_cmd_mit(0.0, 0.0, 1.0, 0.1, 0.0)
    time.sleep(0.3)
    ctrl.disable_all()
    print('ok')
except:
    print('err')
" 2>/dev/null)
    if [ "$RESPONSE" = "ok" ]; then
        echo "  $iface -> arm detected"
        ACTIVE="$ACTIVE $iface"
    else
        echo "  $iface -> no response"
    fi
done

echo $ACTIVE | tr ' ' '\n' | grep -v '^$' > ~/Desktop/openarms/active_cans.txt
echo "Active channels written to active_cans.txt:"
cat ~/Desktop/openarms/active_cans.txt
