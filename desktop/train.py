#!/usr/bin/env python3
"""
Usage (smoke test, tiny model/data, verifies the pipeline works):
    python3 train.py --data-root ./packaged_dataset --output-dir ./checkpoints/smoke_test \
        --epochs 3 --batch-size 2 --chunk-size 20 --limit-episodes 3

Usage (full training run):
    python3 train.py --data-root ./packaged_dataset --output-dir ./checkpoints/act_run1 \
        --epochs 2000 --batch-size 8 --chunk-size 100
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_video import PackagedEpisodeDataset
from model import ACTModel, act_loss


def compute_norm_stats(dataset):
    print("Computing normalization stats...")
    indices = range(0, len(dataset), max(1, len(dataset) // 200))
    states = np.stack([dataset[i]["state"].numpy() for i in tqdm(indices, desc="norm stats")])
    return {"state_mean": states.mean(0).tolist(), "state_std": (states.std(0) + 1e-6).tolist()}


def save_checkpoint(model, output_dir: Path, epoch: int, norm_stats: dict, config: dict, data_root):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "config.json").write_text(json.dumps({**config, "norm_stats": norm_stats}, indent=2))
    (output_dir / "training_provenance.json").write_text(json.dumps({
        "epoch": epoch, "data_root": str(data_root), "policy_type": "act_custom",
    }, indent=2))


def train(args):
    if args.partition:
        from partitioned_dataset import PartitionedDataset
        dataset = PartitionedDataset(args.data_root, args.index_csv, args.partition,
                                      chunk_size=args.chunk_size, arm_role="fr")
    else:
        dataset = PackagedEpisodeDataset(args.data_root, chunk_size=args.chunk_size,
                                          arm_role="fr", limit_episodes=args.limit_episodes,
                                          name_prefix=args.name_prefix, frame_stride=args.frame_stride)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    norm_stats = compute_norm_stats(dataset)
    state_mean = torch.tensor(norm_stats["state_mean"])
    state_std = torch.tensor(norm_stats["state_std"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ACTModel(chunk_size=args.chunk_size, pretrained_backbone=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    config = {"state_dim": 24, "action_dim": 24, "chunk_size": args.chunk_size, "arm_role": "fr"}

    n_batches = len(loader)
    print(f"Starting training: {args.epochs} epochs, {n_batches} batches/epoch, batch_size={args.batch_size}")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            image = batch["image"].to(device)
            state = ((batch["state"] - state_mean) / state_std).to(device)
            action_chunk = batch["action_chunk"].to(device)
            is_pad = batch["is_pad"].to(device)

            pred_actions, mu, logvar = model(image, state, action_chunk, is_pad)
            loss, recon, kl = act_loss(pred_actions, action_chunk, is_pad, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon.item():.4f}",
                "kl": f"{kl.item():.4f}",
            })

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        elapsed = time.time() - epoch_start
        eta_seconds = elapsed * (args.epochs - epoch - 1)
        eta_min = eta_seconds / 60

        print(f"epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f} "
              f"| {elapsed:.1f}s/epoch | ETA {eta_min:.1f} min")

        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, Path(args.output_dir), epoch, norm_stats, config, args.data_root)
            print(f"  -> checkpoint saved to {args.output_dir}")

    print(f"Training complete. Checkpoint at {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--limit-episodes", type=int, default=None, help="For smoke tests: only use N episodes")
    p.add_argument("--index-csv", type=str, default=None)
    p.add_argument("--partition", type=str, default=None)
    p.add_argument("--name-prefix", type=str, default=None)
    p.add_argument("--frame-stride", type=int, default=1)
    args = p.parse_args()
    train(args)