#!/usr/bin/env python3
"""
Runs on the Linux machine. Converts raw collect.py dataset roots into a lightweight
video-encoded package ready for zipping + transfer.

Skips:
  - episodes marked success=False in meta/episodes.jsonl
  - the first 10 episodes (by episode_index) specifically in a root named "dataset"

Usage:
    python3 package_dataset.py \
        --roots ./dataset ./dataset_100 ./dataset_quality \
        --output ./packaged_dataset \
        --status-file ./status.log
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import pandas as pd


def log(status_file: Path, msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(status_file, "a") as f:
        f.write(line + "\n")


def load_success_map(root: Path) -> dict:
    meta_file = root / "meta" / "episodes.jsonl"
    success = {}
    if meta_file.exists():
        with open(meta_file) as f:
            for line in f:
                rec = json.loads(line)
                success[rec["episode_index"]] = rec.get("success", False)
    return success


def encode_episode_video(png_dir: Path, out_mp4: Path, fps: int = 30) -> bool:
    if out_mp4.exists():
        return True
    if not png_dir.exists() or not any(png_dir.glob("*.png")):
        return False
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(png_dir / "%06d.png"),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(out_mp4),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def package(roots, output: Path, status_file: Path):
    status_file.write_text("")  # reset log
    log(status_file, "STARTING packaging job")

    out_data = output / "data"
    out_video = output / "videos"
    out_data.mkdir(parents=True, exist_ok=True)
    out_video.mkdir(parents=True, exist_ok=True)

    manifest = {"episodes": [], "skipped_failed": 0, "skipped_first10": 0, "total_frames": 0}

    for root in roots:
        root = Path(root)
        is_dataset_root = root.name == "dataset"
        success_map = load_success_map(root)

        chunk_dir = root / "data" / "chunk-000"
        episode_files = sorted(chunk_dir.glob("episode_*.parquet"))
        log(status_file, f"Root {root}: found {len(episode_files)} episode parquet files")

        for pq_path in episode_files:
            ep_idx = int(pq_path.stem.split("_")[1])

            if is_dataset_root and ep_idx < 10:
                manifest["skipped_first10"] += 1
                log(status_file, f"  SKIP {root.name}/episode_{ep_idx:06d} (first-10 rule)")
                continue

            if success_map.get(ep_idx, False) is False and ep_idx in success_map:
                manifest["skipped_failed"] += 1
                log(status_file, f"  SKIP {root.name}/episode_{ep_idx:06d} (marked failed)")
                continue

            df = pd.read_parquet(pq_path)
            unique_name = f"{root.name}_episode_{ep_idx:06d}"

            png_dir = root / "tmp_frames" / f"episode_{ep_idx:06d}" / "cam_high"
            out_mp4 = out_video / f"{unique_name}.mp4"
            ok = encode_episode_video(png_dir, out_mp4)

            if not ok:
                log(status_file, f"  WARNING: no chest-cam frames found for {unique_name}, skipping episode")
                continue

            out_pq = out_data / f"{unique_name}.parquet"
            df.to_parquet(out_pq)

            wrist_png_dir = root / "tmp_frames" / f"episode_{ep_idx:06d}" / "cam_right_wrist"
            wrist_mp4 = out_video / f"{unique_name}_wrist.mp4"
            has_wrist = encode_episode_video(wrist_png_dir, wrist_mp4)

            manifest["episodes"].append({
                "name": unique_name,
                "source_root": root.name,
                "source_episode_index": ep_idx,
                "n_frames": len(df),
                "has_wrist_cam": has_wrist,
            })
            manifest["total_frames"] += len(df)
            log(status_file, f"  OK {unique_name}: {len(df)} frames encoded")

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(status_file, f"Packaging complete: {len(manifest['episodes'])} episodes, "
                      f"{manifest['total_frames']} frames, "
                      f"{manifest['skipped_failed']} failed skipped, "
                      f"{manifest['skipped_first10']} first-10 skipped")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--status-file", type=Path, default=Path("./status.log"))
    args = p.parse_args()
    package(args.roots, args.output, args.status_file)