#!/usr/bin/env python3
"""
Video-backed dataset loader for the packaged format produced by
package_dataset.py. Reads MP4 frames via OpenCV instead of PNGs.
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
                 limit_episodes=None, name_prefix=None, frame_stride=1):
        self.root = Path(root)
        self.chunk_size = chunk_size
        self.arm_role = arm_role
        self.image_size = image_size

        manifest = json.loads((self.root / "manifest.json").read_text())
        episodes = manifest["episodes"]

        if name_prefix:
            episodes = [e for e in episodes if e["name"].startswith(name_prefix)]

        if limit_episodes:
            episodes = episodes[:limit_episodes]

        self.samples = []
        self.episode_data = {}

        for ep in episodes:
            name = ep["name"]
            pq_path = self.root / "data" / f"{name}.parquet"
            df = pd.read_parquet(pq_path).sort_values("frame_index").reset_index(drop=True)
            self.episode_data[name] = df
            for frame_idx in range(0, len(df), frame_stride):
                self.samples.append((name, frame_idx, len(df)))

        print(f"Loaded {len(episodes)} episodes, {len(self.samples)} frames total.")

    def __len__(self):
        return len(self.samples)

    def _get_frame(self, name: str, frame_idx: int) -> np.ndarray:
        mp4_path = self.root / "videos" / f"{name}.mp4"
        cap = cv2.VideoCapture(str(mp4_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return np.zeros((*self.image_size, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(frame, (self.image_size[1], self.image_size[0]))

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
        image = self._get_frame(name, frame_idx)
        state = self._extract_state(row)

        end = min(frame_idx + self.chunk_size, n)
        action_chunk = np.stack([self._extract_action(df.iloc[i]) for i in range(frame_idx, end)])
        pad_len = self.chunk_size - action_chunk.shape[0]
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        if pad_len > 0:
            action_chunk = np.pad(action_chunk, ((0, pad_len), (0, 0)), mode="edge")
            is_pad[-pad_len:] = True

        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            "state": torch.from_numpy(state),
            "action_chunk": torch.from_numpy(action_chunk),
            "is_pad": torch.from_numpy(is_pad),
        }