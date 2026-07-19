#!/usr/bin/env python3
"""
Extends PackagedEpisodeDataset to filter by a named partition in
episode_index.csv, rather than hardcoding roots/limits.
"""

import pandas as pd
from pathlib import Path

from dataset_video import PackagedEpisodeDataset


def get_partition_episodes(index_csv: str, partition_name: str) -> list[tuple[str, int]]:
    df = pd.read_csv(index_csv)
    sel = df[(df["partition"] == partition_name) & (df["include"] == True)]
    if len(sel) == 0:
        raise ValueError(f"No episodes found for partition \\'{partition_name}\\' in {index_csv}")
    return list(zip(sel["root"], sel["episode_index"]))


class PartitionedDataset(PackagedEpisodeDataset):
    """
    Same as PackagedEpisodeDataset, but restricts to episodes listed
    under a specific partition name in episode_index.csv.
    Requires that package_dataset.py's manifest.json "name" field
    follows the "<root>_episode_<idx:06d>" convention already used.
    """
    def __init__(self, packaged_root, index_csv, partition_name, chunk_size=100, arm_role="fr",
                 use_wrist_cam=False):
        wanted = get_partition_episodes(index_csv, partition_name)
        wanted_names = {f"{root}_episode_{idx:06d}" for root, idx in wanted}

        super().__init__(packaged_root, chunk_size=chunk_size, arm_role=arm_role, limit_episodes=None,
                          use_wrist_cam=use_wrist_cam)

        self.samples = [s for s in self.samples if s[0] in wanted_names]
        self.episode_data = {k: v for k, v in self.episode_data.items() if k in wanted_names}
        self.has_wrist = {k: v for k, v in self.has_wrist.items() if k in wanted_names}
        print(f"Partition \\'{partition_name}\\': {len(wanted_names)} episodes, {len(self.samples)} frames")