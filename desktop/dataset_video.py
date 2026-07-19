#!/usr/bin/env python3
"""
Video-backed dataset loader for the packaged format produced by
package_dataset.py + encode_wrist_videos.py. Reads MP4 frames via OpenCV.

Supports both single-camera (chest only) and dual-camera (chest + wrist)
episodes. Falls back to zero-image for wrist if an episode has no wrist
video (has_wrist_cam=False in manifest.json), so old episodes without
wrist footage still train fine alongside new dual-camera episodes.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PackagedEpisodeDataset(Dataset):
    def __init__(self, root, chunk_size=100, arm_role="fr", image_size=(480, 848),
                 limit_episodes=None, name_prefix=None, frame_stride=1,
                 use_wrist_cam=True):
        self.root = Path(root)
        self.chunk_size = chunk_size
        self.arm_role = arm_role
        self.image_size = image_size
        self.use_wrist_cam = use_wrist_cam

        manifest = json.loads((self.root / "manifest.json").read_text())
        episodes = manifest["episodes"]

        if name_prefix:
            episodes = [e for e in episodes if e["name"].startswith(name_prefix)]

        if limit_episodes:
            episodes = episodes[:limit_episodes]

        self.samples = []
        self.episode_data = {}
        self.has_wrist = {}

        for ep in episodes:
            name = ep["name"]
            pq_path = self.root / "data" / f"{name}.parquet"
            df = pd.read_parquet(pq_path).sort_values("frame_index").reset_index(drop=True)
            self.episode_data[name] = df
            self.has_wrist[name] = bool(ep.get("has_wrist_cam", False))
            for frame_idx in range(0, len(df), frame_stride):
                self.samples.append((name, frame_idx, len(df)))

        n_with_wrist = sum(1 for v in self.has_wrist.values() if v)
        print(f"Loaded {len(episodes)} episodes, {len(self.samples)} frames total. "
              f"{n_with_wrist}/{len(episodes)} episodes have wrist-cam video.")

    def __len__(self):
        return len(self.samples)

    def _get_frame(self, mp4_path: Path, frame_idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(mp4_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return np.zeros((*self.image_size, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(frame, (self.image_size[1], self.image_size[0]))

    def _get_images(self, name: str, frame_idx: int) -> dict:
        chest_path = self.root / "videos" / f"{name}.mp4"
        chest_img = self._get_frame(chest_path, frame_idx)

        if not self.use_wrist_cam:
            return {"cam_high": chest_img}

        if self.has_wrist.get(name, False):
            wrist_path = self.root / "videos" / f"{name}_wrist.mp4"
            wrist_img = self._get_frame(wrist_path, frame_idx)
        else:
            wrist_img = np.zeros((*self.image_size, 3), dtype=np.uint8)

        return {"cam_high": chest_img, "cam_wrist": wrist_img}

    def _extract_state(self, row) -> np.ndarray:
        state = np.asarray(row["observation.state"], dtype=np.float32)
        return state[24:48] if self.arm_role == "fr" else state[0:24]

    def _extract_action(self, row) -> np.ndarray:
        action = np.asarray(row["action"], dtype=np.float32)
        half = action[8:16] if self.arm_role == "fr" else action[0:8]
        vec = np.zeros(24, dtype=np.float32)
        vec[0::3] = half
        return vec

    def __getitem__(self, idx):
        name, frame_idx, n = self.samples[idx]
        df = self.episode_data[name]

        row = df.iloc[frame_idx]
        images = self._get_images(name, frame_idx)
        state = self._extract_state(row)

        end = min(frame_idx + self.chunk_size, n)
        action_chunk = np.stack([self._extract_action(df.iloc[i]) for i in range(frame_idx, end)])
        pad_len = self.chunk_size - action_chunk.shape[0]
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        if pad_len > 0:
            action_chunk = np.pad(action_chunk, ((0, pad_len), (0, 0)), mode="edge")
            is_pad[-pad_len:] = True

        sample = {
            "image": torch.from_numpy(images["cam_high"]).permute(2, 0, 1).float() / 255.0,
            "state": torch.from_numpy(state),
            "action_chunk": torch.from_numpy(action_chunk),
            "is_pad": torch.from_numpy(is_pad),
        }

        if self.use_wrist_cam:
            sample["wrist_image"] = torch.from_numpy(images["cam_wrist"]).permute(2, 0, 1).float() / 255.0
            sample["has_wrist"] = torch.tensor(self.has_wrist.get(name, False))

        return sample