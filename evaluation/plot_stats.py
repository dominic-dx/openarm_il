#!/usr/bin/env python3
"""
Plots per-frame stats produced by analyze_dataset_attention.py
(outputs/dataset_analysis/per_frame_stats.csv).

Generates, per episode:
  1. traj_error vs frame index -- spot where in the episode the model
     struggles (e.g. grasp/release transitions vs steady reaching motion)
  2. state_share and latent_share vs frame index, on the same axes --
     see if reliance on proprioception vs vision shifts over the episode
     (e.g. spikes right before contact, when vision might get occluded
     by the gripper itself)
  3. A traj_error histogram, to see if the mean is being dragged up by
     a few outlier frames (spikes) or if error is uniformly elevated
     throughout

If per_frame_stats.csv contains multiple episodes, each gets its own set
of plots, saved as separate PNGs, plus one combined overlay plot comparing
traj_error across all episodes on shared axes.

Usage:
    python3 plot_dataset_analysis.py \
        --csv ./outputs/dataset_analysis/per_frame_stats.csv \
        --out-dir ./outputs/dataset_analysis/plots
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_episode(df_ep, ep_name, out_dir: Path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)

    # 1. traj_error over time
    axes[0].plot(df_ep["frame"], df_ep["traj_error"], color="crimson", linewidth=1)
    axes[0].axhline(df_ep["traj_error"].mean(), color="black", linestyle="--", linewidth=1,
                     label=f"mean={df_ep['traj_error'].mean():.4f}")
    axes[0].set_title(f"{ep_name} -- trajectory error over episode")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("traj_error (MAE)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. state_share / latent_share over time
    axes[1].plot(df_ep["frame"], df_ep["state_share"], color="steelblue", linewidth=1, label="state_share")
    axes[1].plot(df_ep["frame"], df_ep["latent_share"], color="darkorange", linewidth=1, label="latent_share")
    axes[1].set_title(f"{ep_name} -- attention share on state/latent tokens over episode")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("attention share")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 3. traj_error histogram
    axes[2].hist(df_ep["traj_error"], bins=40, color="slateblue", edgecolor="black", alpha=0.8)
    axes[2].axvline(df_ep["traj_error"].mean(), color="black", linestyle="--", linewidth=1,
                     label=f"mean={df_ep['traj_error'].mean():.4f}")
    axes[2].axvline(df_ep["traj_error"].median(), color="green", linestyle="--", linewidth=1,
                     label=f"median={df_ep['traj_error'].median():.4f}")
    axes[2].set_title(f"{ep_name} -- trajectory error distribution")
    axes[2].set_xlabel("traj_error (MAE)")
    axes[2].set_ylabel("frame count")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"{ep_name}_stats.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_combined(df, out_dir: Path):
    episodes = df["episode"].unique()
    if len(episodes) < 2:
        return  # nothing to compare

    fig, ax = plt.subplots(figsize=(10, 5))
    for ep in episodes:
        sub = df[df["episode"] == ep]
        ax.plot(sub["frame"], sub["traj_error"], linewidth=1, alpha=0.8, label=ep)
    ax.set_title("trajectory error across episodes")
    ax.set_xlabel("frame")
    ax.set_ylabel("traj_error (MAE)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "combined_traj_error.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")

    # Also a bar chart of per-episode mean traj_error, for a quick ranking
    summary = df.groupby("episode")["traj_error"].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(summary.index, summary.values, color="crimson", alpha=0.8)
    ax2.set_title("mean trajectory error per episode")
    ax2.set_ylabel("mean traj_error (MAE)")
    ax2.tick_params(axis="x", rotation=45, labelsize=7)
    fig2.tight_layout()
    out_path2 = out_dir / "combined_mean_traj_error_bar.png"
    fig2.savefig(out_path2, dpi=120)
    plt.close(fig2)
    print(f"Saved {out_path2}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("./outputs/dataset_analysis/per_frame_stats.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("./outputs/dataset_analysis/plots"))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ep_name, df_ep in df.groupby("episode"):
        plot_episode(df_ep.sort_values("frame"), ep_name, args.out_dir)

    plot_combined(df, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()