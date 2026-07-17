#!/usr/bin/env python3
"""
OpenArm Data Collection Server
Run: python3 collect.py
Access: http://localhost:5000
"""

import os
import time
import json
import threading
import queue
import subprocess
from pathlib import Path
from flask import Flask, Response, jsonify, request
import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import can
from damiao_motor import DaMiaoController

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/oalami/Desktop/openarms/dataset_100")
ARM_MAPPING_FILE = "/home/oalami/Desktop/openarms/arm_mapping.txt"
MOTOR_IDS = range(1, 9)
CONTROL_FREQ = 50
DT = 1.0 / CONTROL_FREQ

CAMERA_SERIALS = {
    "cam_high":       "254622073959",
    "cam_left_wrist": "352122272724",
    "cam_right_wrist": "352122273783",
}
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30
PNG_THREADS_PER_CAM = 4

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
def get_next_episode_index():
    p = DATASET_ROOT / "data" / "chunk-000"
    if not p.exists():
        return 0
    existing = list(p.glob("episode_*.parquet"))
    if not existing:
        return 0
    return max(int(f.stem.split("_")[1]) for f in existing) + 1
state = {
    "recording": False,
    "episode_index": get_next_episode_index(),
    "frame_count": 0,
    "task": "place cube in flask",
    "status": "idle",
    "motors_ok": False,
    "episode_start_time": 0.0,
}
motors = {}
motor_channels = {}
latest_frames = {}
episode_buffer = []
lock = threading.Lock()
png_queues = {k: queue.Queue() for k in CAMERA_SERIALS}
cam_frame_counters = {k: 0 for k in CAMERA_SERIALS}

# ── Path helpers ──────────────────────────────────────────────────────────────
def parquet_path(idx):
    p = DATASET_ROOT / "data" / "chunk-000"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"episode_{idx:06d}.parquet"

def video_path(cam_key, idx):
    p = DATASET_ROOT / "videos" / "chunk-000" / f"observation.images.{cam_key}"
    p.mkdir(parents=True, exist_ok=True)
    return str(p / f"episode_{idx:06d}.mp4")

def png_dir(cam_key, ep_idx):
    p = DATASET_ROOT / "tmp_frames" / f"episode_{ep_idx:06d}" / cam_key
    p.mkdir(parents=True, exist_ok=True)
    return p

# ── PNG writer workers ────────────────────────────────────────────────────────
def png_writer_worker(cam_key, q):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        ep_idx, frame_idx, frame = item
        path = png_dir(cam_key, ep_idx) / f"{frame_idx:06d}.png"
        cv2.imwrite(str(path), frame)
        q.task_done()

for cam_key in CAMERA_SERIALS:
    for _ in range(PNG_THREADS_PER_CAM):
        t = threading.Thread(target=png_writer_worker, args=(cam_key, png_queues[cam_key]), daemon=True)
        t.start()

# ── FFmpeg encode ─────────────────────────────────────────────────────────────
def encode_video(cam_key, ep_idx):
    src_dir = png_dir(cam_key, ep_idx)
    dst = video_path(cam_key, ep_idx)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(CAM_FPS),
        "-i", str(src_dir / "%06d.png"),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        dst
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"FFmpeg error for {cam_key}: {result.stderr.decode()}")
        return False
    for f in src_dir.glob("*.png"):
        f.unlink()
    src_dir.rmdir()
    return True

def encode_all_videos(ep_idx):
    threads = []
    for cam_key in CAMERA_SERIALS:
        t = threading.Thread(target=encode_video, args=(cam_key, ep_idx), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

# ── Dataset helpers ───────────────────────────────────────────────────────────
def save_episode(rows, success):
    if not rows:
        return
    idx = rows[0]["episode_index"]
    for r in rows:
        r["success"] = success
        r["next.done"] = False
    rows[-1]["next.done"] = True
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df), parquet_path(idx))
    meta_dir = DATASET_ROOT / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "episodes.jsonl", "a") as f:
        f.write(json.dumps({
            "episode_index": idx,
            "task": rows[0]["task"],
            "length": len(rows),
            "success": success,
        }) + "\n")

# ── Motor helpers ─────────────────────────────────────────────────────────────
def load_mapping():
    mapping = {}
    with open(ARM_MAPPING_FILE) as f:
        for line in f:
            if "=" in line:
                role, ch = line.strip().split("=")
                mapping[role] = ch
    return mapping

def clear_faults(channel):
    try:
        bus = can.Bus(interface="socketcan", channel=channel, bitrate=1000000)
        for mid in range(1, 9):
            bus.send(can.Message(arbitration_id=mid,
                                 data=[0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFB],
                                 is_extended_id=False))
            time.sleep(0.05)
        bus.shutdown()
        time.sleep(0.5)
    except Exception as e:
        print(f"Fault clear error on {channel}: {e}")

def setup_motors():
    global motors, motor_channels
    mapping = load_mapping()
    for role, ch in mapping.items():
        motor_channels[role] = ch
        clear_faults(ch)
        ctrl = DaMiaoController(channel=ch, bustype="socketcan")
        arm_motors = {}
        for i in MOTOR_IDS:
            arm_motors[i] = ctrl.add_motor(motor_id=i, feedback_id=0x10+i, motor_type="4310")
        for m in arm_motors.values():
            m.enable()
            time.sleep(0.05)
        time.sleep(0.5)
        motors[role] = (ctrl, arm_motors)
    state["motors_ok"] = True
    print("Motors OK.")

def get_arm_state(arm_motors):
    """Query leader arm state — sends zero-gain MIT to trigger feedback without resisting motion."""
    state_vec = []
    positions = []
    for i in MOTOR_IDS:
        # kp=0 kd=0 means zero torque — motor is backdrivable, just triggers feedback packet
        arm_motors[i].send_cmd_mit(0.0, 0.0, 0.0, 0.0, 0.0)
        s = arm_motors[i].get_states()
        state_vec.extend([s.get("pos", 0.0), s.get("vel", 0.0), s.get("torq", 0.0)])
        positions.append(s.get("pos", 0.0))
    return state_vec, positions

def get_arm_state_passive(arm_motors):
    """Read last known state without sending any command — use for follower arms."""
    state_vec = []
    positions = []
    for i in MOTOR_IDS:
        s = arm_motors[i].get_states()
        state_vec.extend([s.get("pos", 0.0), s.get("vel", 0.0), s.get("torq", 0.0)])
        positions.append(s.get("pos", 0.0))
    return state_vec, positions

def disable_all_motors():
    for role, ch in motor_channels.items():
        try:
            ctrl = DaMiaoController(channel=ch, bustype="socketcan")
            for i in range(1, 9):
                ctrl.add_motor(motor_id=i, feedback_id=0x10+i, motor_type="4310")
            ctrl.disable_all()
            print(f"{ch} disabled")
        except Exception as e:
            print(f"{ch} disable ERR: {e}")

# ── Camera setup ──────────────────────────────────────────────────────────────
pipelines = {}

def setup_single_camera(cam_key, serial):
    try:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.bgr8, CAM_FPS)
        pipeline.start(cfg)
        pipelines[cam_key] = pipeline
        for _ in range(10):
            try:
                pipeline.wait_for_frames(timeout_ms=5000)
                break
            except Exception:
                time.sleep(0.5)
        print(f"  {cam_key} ready.")
    except Exception as e:
        print(f"  {cam_key} FAILED: {e}")

def setup_cameras():
    threads = []
    for cam_key, serial in CAMERA_SERIALS.items():
        t = threading.Thread(target=setup_single_camera, args=(cam_key, serial))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("Cameras warmed up.")

def camera_thread(cam_key, pipeline):
    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=500)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            _, jpeg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 40])
            with lock:
                latest_frames[cam_key] = jpeg.tobytes()
                if state["recording"]:
                    ep_idx = state["episode_index"]
                    frame_idx = cam_frame_counters[cam_key]
                    cam_frame_counters[cam_key] += 1
            if state["recording"]:
                try:
                    png_queues[cam_key].put_nowait((ep_idx, frame_idx, color.copy()))
                except queue.Full:
                    print(f"WARNING: PNG queue full for {cam_key}")
        except Exception:
            pass

def camera_loop():
    for cam_key, pipeline in pipelines.items():
        t = threading.Thread(target=camera_thread, args=(cam_key, pipeline), daemon=True)
        t.start()

# ── Teleop ────────────────────────────────────────────────────────────────────
SCALE = 1.5
ll_init = {}
lr_init = {}
fl_init = {}
fr_init = {}

def teleop_loop():
    global ll_init, lr_init, fl_init, fr_init
    while not state["motors_ok"]:
        time.sleep(0.1)
    _, ll_motors = motors.get("ll", (None, {}))
    _, lr_motors = motors.get("lr", (None, {}))
    _, fl_motors = motors.get("fl", (None, {}))
    _, fr_motors = motors.get("fr", (None, {}))
    if not all([ll_motors, lr_motors, fl_motors, fr_motors]):
        print("Teleop: missing arm mapping.")
        return
    print("waiting for motors to settle")
    time.sleep(3.0)
    _, ll_p = get_arm_state(ll_motors)
    _, lr_p = get_arm_state(lr_motors)
    _, fl_p = get_arm_state(fl_motors)
    _, fr_p = get_arm_state(fr_motors)
    ll_init = {i+1: ll_p[i] for i in range(8)}
    lr_init = {i+1: lr_p[i] for i in range(8)}
    fl_init = {i+1: fl_p[i] for i in range(8)}
    fr_init = {i+1: fr_p[i] for i in range(8)}
    print("Teleop running.")
    while True:
        t0 = time.time()
        try:
            ll_sv, ll_pos = get_arm_state(ll_motors)
            lr_sv, lr_pos = get_arm_state(lr_motors)
            fl_actions = []
            fr_actions = []
            for i in MOTOR_IDS:
                ld = (ll_pos[i-1] - ll_init[i]) * SCALE
                rd = (lr_pos[i-1] - lr_init[i]) * SCALE
                if i == 8:
                    ld *= -3
                    rd *= 5.5
                fl_t = fl_init[i] + ld
                fr_t = fr_init[i] + rd
                fl_motors[i].send_cmd_mit(fl_t, 0.0, 10.0, 1.0, 0.0)
                fr_motors[i].send_cmd_mit(fr_t, 0.0, 10.0, 1.0, 0.0)
                fl_actions.append(fl_t)
                fr_actions.append(fr_t)
            # Read actual follower state passively — no CAN command sent, no jitter
            fl_sv, _ = get_arm_state_passive(fl_motors)
            fr_sv, _ = get_arm_state_passive(fr_motors)
            with lock:
                if state["recording"]:
                    row = {
                        "timestamp":         time.time() - state["episode_start_time"],
                        "frame_index":       state["frame_count"],
                        "episode_index":     state["episode_index"],
                        "observation.state": fl_sv + fr_sv,
                        "action":            fl_actions + fr_actions,
                        "task":              state["task"],
                        "next.done":         False,
                        "success":           False,
                    }
                    episode_buffer.append(row)
                    state["frame_count"] += 1
        except Exception as e:
            print(f"Teleop error: {e}")
        time.sleep(max(0, DT - (time.time() - t0)))

# ── HTML ──────────────────────────────────────────────────────────────────────

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/stats")
def stats():
    meta_file = DATASET_ROOT / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return jsonify({"total": 0, "success": 0, "fail": 0})
    success = fail = 0
    with open(meta_file) as f:
        for line in f:
            ep = json.loads(line)
            if ep.get("success"):
                success += 1
            else:
                fail += 1
    return jsonify({"total": success + fail, "success": success, "fail": fail})



@app.route("/")
def index():
    with open("/home/oalami/Desktop/openarms/index.html") as f:
        return f.read()

@app.route("/feed/<cam_key>")
def feed(cam_key):
    with lock:
        frame = latest_frames.get(cam_key)
    if frame is None:
        return Response(status=404)
    return Response(frame, mimetype="image/jpeg")

def generate_stream(cam_key):
    while True:
        with lock:
            frame = latest_frames.get(cam_key)
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.033)

@app.route("/stream/<cam_key>")
def stream(cam_key):
    return Response(generate_stream(cam_key),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def get_status():
    with lock:
        return jsonify({
            "recording":     state["recording"],
            "episode_index": state["episode_index"],
            "frame_count":   state["frame_count"],
            "status":        state["status"],
            "motors_ok":     state["motors_ok"],
        })

@app.route("/set_task", methods=["POST"])
def set_task():
    data = request.get_json()
    state["task"] = data.get("task", state["task"])
    return jsonify({"task": state["task"]})

@app.route("/start_episode", methods=["POST"])
def start_episode():
    with lock:
        if state["recording"]:
            return jsonify({"error": "already recording"}), 400
        episode_buffer.clear()
        state["frame_count"] = 0
        state["recording"] = True
        state["status"] = "recording"
        state["episode_start_time"] = time.time()
        idx = state["episode_index"]
        for k in cam_frame_counters:
            cam_frame_counters[k] = 0
        for cam_key in CAMERA_SERIALS:
            png_dir(cam_key, idx)
    return jsonify({"episode_index": idx})

@app.route("/stop_episode", methods=["POST"])
def stop_episode():
    data = request.get_json()
    success = data.get("success", False)
    with lock:
        if not state["recording"]:
            return jsonify({"error": "not recording"}), 400
        state["recording"] = False
        state["status"] = "encoding"
        rows = list(episode_buffer)
        idx = state["episode_index"]
        steps = state["frame_count"]
    for q in png_queues.values():
        q.join()
    save_episode(rows, success)
    #encode_all_videos(idx)
    with lock:
        state["episode_index"] += 1
        state["status"] = "idle"
    return jsonify({"episode_index": idx, "steps": steps, "success": success})

@app.route("/check_episode/<int:idx>")
def check_episode(idx):
    results = {}
    pq_path = parquet_path(idx)
    if pq_path.exists():
        try:
            df = pd.read_parquet(pq_path)
            results["parquet"] = {
                "ok": len(df) > 0,
                "rows": len(df),
                "duration_s": round(float(df["timestamp"].max() - df["timestamp"].min()), 2) if len(df) > 1 else 0,
            }
        except Exception as e:
            results["parquet"] = {"ok": False, "error": str(e)}
    else:
        results["parquet"] = {"ok": False, "error": "file not found"}

    results["frames"] = {}
    for cam_key in CAMERA_SERIALS:
        png_count = len(list(png_dir(cam_key, idx).glob("*.png")))
        results["frames"][cam_key] = png_count

    results["all_ok"] = results["parquet"]["ok"] and all(v > 0 for v in results["frames"].values())

    return jsonify(results)

@app.route("/skip_episode", methods=["POST"])
def skip_episode():
    with lock:
        if not state["recording"]:
            return jsonify({"error": "not recording"}), 400
        state["recording"] = False
        state["status"] = "idle"
        idx = state["episode_index"]
    # Clear PNG queues
    for q in png_queues.values():
        try:
            while True:
                q.get_nowait()
                q.task_done()
        except:
            pass
    # Delete any PNGs already written
    import shutil
    for cam_key in CAMERA_SERIALS:
        d = png_dir(cam_key, idx)
        if d.exists():
            shutil.rmtree(d)
    # Don't increment episode_index — reuse this index
    return jsonify({"episode_index": idx})


@app.route("/estop", methods=["POST"])
def estop():
    with lock:
        state["recording"] = False
        state["status"] = "estop"
        state["motors_ok"] = False
    disable_all_motors()
    return jsonify({"ok": True})

# ── Startup ───────────────────────────────────────────────────────────────────
def startup():
    print("Setting up cameras...")
    #setup_cameras()
    print("Setting up motors...")
    try:
        setup_motors()
    except Exception as e:
        print(f"Motor setup failed: {e}")
    setup_cameras()
    camera_loop()
    threading.Thread(target=teleop_loop, daemon=True).start()
    print("Startup complete.")

if __name__ == "__main__":
    import os
    os.system("fuser -k 5000/tcp 2>/dev/null")
    #os.system("pkill -f 'collect.py' 2>/dev/null")
    import time
    time.sleep(1)
    threading.Thread(target=startup, daemon=True).start()
    print("Starting server at http://0.0.0.0:5000")
    app.run(host="0.0.0.0",   port=5000, threaded=True)
