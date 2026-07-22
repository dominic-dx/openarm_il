#!/usr/bin/env python3
"""
Offline dataset-driven ACT validation + attention visualization.

Two independent validation signals per episode, computed frame-by-frame:
  1. TRAJECTORY VALIDATION: at each frame, run the model on the real
     recorded (image, state) and compare its predicted action_chunk[0]
     (immediate next action) against the actually-recorded next action.
     Low error here means the model's predictions track real teleop
     behavior on data it may have already seen in training -- a sanity
     check, not proof of generalization, but a necessary baseline: if it
     can't match training-distribution trajectories, live rollout failure
     is not surprising.
  2. ATTENTION VALIDATION: reuses the same cross-attention extraction as
     live_attention_viewer.py / batch_attention_map.py, overlaid on every
     frame, encoded into an output video. Watching this back tells you
     whether attention meaningfully tracks the moving gripper/object
     across a real episode, or stays diffuse/fixed on background --
     exactly the same diagnostic as the live viewer, but replayable and
     shareable, and directly time-synced to the actual recorded motion.

These are independent checks: a model could nail #1 (good trajectory
match on data it memorized) while failing #2 (never actually "looking"
at anything -- e.g. relying almost entirely on the state token, matching
recorded state->action patterns without real visual grounding). Seeing
both together is the point.

Assumes the packaged dataset format (manifest.json + per-episode parquet
+ mp4), matching PackagedEpisodeDataset in dataset_video.py -- this is
the format actually consumed by train.py, so trajectory error here is
directly comparable to training loss.

Usage (smoke test on 1 episode first, as requested):
    python3 analyze_dataset_attention.py \
        --checkpoint ./checkpoints/model5/epoch_7 \
        --data-root ./packaged_dataset \
        --max-episodes 1 \
        --out-dir ./outputs/dataset_analysis

Then scale up:
    python3 analyze_dataset_attention.py \
        --checkpoint ./checkpoints/model5/epoch_7 \
        --data-root ./packaged_dataset \
        --max-episodes 20 \
        --out-dir ./outputs/dataset_analysis
"""

import argparse
import json
import csv
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F

from model import ACTModel

IMAGE_SIZE = (480, 848)  # H, W -- matches training preprocessing


# ── Manual decoder pass to expose cross-attention weights ──────────────────
def manual_decoder_forward_with_attn(decoder, tgt, memory):
    x = tgt
    cross_attn_weights_per_layer = []
    for layer in decoder.layers:
        sa_out, _ = layer.self_attn(x, x, x, need_weights=False)
        x = layer.norm1(x + layer.dropout1(sa_out))
        ca_out, ca_weights = layer.multihead_attn(
            x, memory, memory, need_weights=True, average_attn_weights=False
        )
        cross_attn_weights_per_layer.append(ca_weights.detach())
        x = layer.norm2(x + layer.dropout2(ca_out))
        ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = layer.norm3(x + layer.dropout3(ff_out))
    return x, torch.stack(cross_attn_weights_per_layer, dim=0)


_grid_size_cache = {}


def get_backbone_grid_size(backbone_module, image_hw, cache_key):
    if cache_key in _grid_size_cache:
        return _grid_size_cache[cache_key]
    with torch.no_grad():
        dummy = torch.zeros(1, 3, *image_hw, device=next(backbone_module.parameters()).device)
        feat = backbone_module(dummy)
    result = (feat.shape[-2], feat.shape[-1])
    _grid_size_cache[cache_key] = result
    return result


@torch.no_grad()
def run_model_with_attention(model, image, state, wrist_image=None):
    """Returns pred_actions (chunk) AND cross-attention maps in one pass,
    so trajectory validation and attention validation share the same
    forward pass -- no redundant inference."""
    B = image.shape[0]
    img_tokens = model.encode_image(image)
    N_img = img_tokens.shape[1]

    tokens_list = [img_tokens]
    N_wrist = 0
    if wrist_image is not None:
        wrist_feat = model.wrist_backbone(wrist_image)
        wrist_feat = model.wrist_backbone_proj(wrist_feat)
        wrist_tokens = wrist_feat.flatten(2).permute(0, 2, 1)
        N_wrist = wrist_tokens.shape[1]
        tokens_list.append(wrist_tokens)

    state_token = model.state_proj(state).unsqueeze(1)
    z = torch.zeros(B, model.latent_dim, device=image.device)
    latent_token = model.latent_proj(z).unsqueeze(1)
    tokens_list.extend([state_token, latent_token])
    memory = torch.cat(tokens_list, dim=1)

    queries = model.query_embed.unsqueeze(0).expand(B, -1, -1)
    decoded, attn_weights = manual_decoder_forward_with_attn(model.decoder, queries, memory)
    pred_actions = model.action_head(decoded)  # (B, T, action_dim)

    avg_weights = attn_weights.mean(dim=(0, 2))  # (B, T, S)
    per_query = avg_weights[0].mean(dim=0)

    idx = 0
    main_attn = per_query[idx:idx + N_img]; idx += N_img
    wrist_attn = None
    if wrist_image is not None:
        wrist_attn = per_query[idx:idx + N_wrist]; idx += N_wrist
    state_attn = per_query[idx]; idx += 1
    latent_attn = per_query[idx]

    Hp, Wp = get_backbone_grid_size(model.backbone, image.shape[-2:], "main")
    main_grid = (main_attn / (main_attn.sum() + 1e-8)).reshape(Hp, Wp).cpu().numpy()

    result = {"main": main_grid, "state_share": float(state_attn), "latent_share": float(latent_attn),
              "pred_actions": pred_actions}
    if wrist_attn is not None:
        Hpw, Wpw = get_backbone_grid_size(model.wrist_backbone, wrist_image.shape[-2:], "wrist")
        result["wrist"] = (wrist_attn / (wrist_attn.sum() + 1e-8)).reshape(Hpw, Wpw).cpu().numpy()
    return result


def make_heatmap_overlay_bgr(rgb_image: np.ndarray, attn_grid: np.ndarray, alpha=0.5) -> np.ndarray:
    H, W = rgb_image.shape[:2]
    grid_t = torch.from_numpy(attn_grid).float().unsqueeze(0).unsqueeze(0)
    heat = F.interpolate(grid_t, size=(H, W), mode="bilinear", align_corners=False)
    heat = heat.squeeze().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_u8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(base_bgr, 1 - alpha, heat_color, alpha, 0)


def label_frame(frame_bgr, lines):
    out = frame_bgr.copy()
    for i, text in enumerate(lines):
        y = 25 + i * 25
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def load_model(checkpoint_dir: Path, device):
    config = json.loads((checkpoint_dir / "config.json").read_text())
    model = ACTModel(
        state_dim=config.get("state_dim", 24),
        action_dim=config.get("action_dim", 24),
        chunk_size=config.get("chunk_size", 100),
        pretrained_backbone=False,
        use_wrist_cam=config.get("use_wrist_cam", False),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))
    model.eval()
    return model, config


def get_frame(mp4_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, (IMAGE_SIZE[1], IMAGE_SIZE[0]))


def extract_state(row, arm_role="fr") -> np.ndarray:
    state = np.asarray(row["observation.state"], dtype=np.float32)
    return state[24:48] if arm_role == "fr" else state[0:24]


def extract_next_action_positions(row, arm_role="fr") -> np.ndarray:
    """Actual recorded action (positions only, 8-dim) for the immediate next step."""
    action = np.asarray(row["action"], dtype=np.float32)
    return action[8:16] if arm_role == "fr" else action[0:8]


def analyze_episode(model, config, ep_name, root: Path, out_dir: Path, fps=15):
    manifest = json.loads((root / "manifest.json").read_text())
    ep_info = next((e for e in manifest["episodes"] if e["name"] == ep_name), None)
    if ep_info is None:
        raise ValueError(f"Episode {ep_name} not found in manifest.json")

    df = pd.read_parquet(root / "data" / f"{ep_name}.parquet").sort_values("frame_index").reset_index(drop=True)
    chest_path = root / "videos" / f"{ep_name}.mp4"
    use_wrist_cam = config.get("use_wrist_cam", False) and ep_info.get("has_wrist_cam", False)
    wrist_path = root / "videos" / f"{ep_name}_wrist.mp4" if use_wrist_cam else None

    device = next(model.parameters()).device
    state_mean = torch.tensor(config["norm_stats"]["state_mean"], device=device)
    state_std = torch.tensor(config["norm_stats"]["state_std"], device=device)

    ep_out_dir = out_dir / ep_name
    frames_dir = ep_out_dir / "tmp_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    per_frame_rows = []
    n = len(df)
    for frame_idx in range(n - 1):  # -1 so a "next action" always exists to compare against
        row = df.iloc[frame_idx]
        chest_img = get_frame(chest_path, frame_idx)
        wrist_img = get_frame(wrist_path, frame_idx) if use_wrist_cam else None

        state_np = extract_state(row)
        image_t = torch.from_numpy(chest_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        state_norm = (torch.from_numpy(state_np).to(device) - state_mean) / state_std
        state_t = state_norm.unsqueeze(0)
        wrist_t = None
        if wrist_img is not None:
            wrist_t = torch.from_numpy(wrist_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        result = run_model_with_attention(model, image_t, state_t, wrist_image=wrist_t)

        # ---- Validation 1: trajectory ----
        pred_next_full = result["pred_actions"][0, 0].cpu().numpy()  # (24,) interleaved pos slots
        pred_next_pos = pred_next_full[0::3]  # matches dataset_video.py's vec[0::3]=half packing
        actual_next_pos = extract_next_action_positions(row)
        traj_error = float(np.abs(pred_next_pos - actual_next_pos).mean())

        # ---- Validation 2: attention overlay frame ----
        chest_overlay = make_heatmap_overlay_bgr(chest_img, result["main"])
        chest_overlay = label_frame(chest_overlay, [
            f"frame {frame_idx}  traj_err={traj_error:.4f}",
            f"state_share={result['state_share']:.3f}  latent_share={result['latent_share']:.3f}",
        ])
        panels = [chest_overlay]
        if "wrist" in result:
            wrist_overlay = make_heatmap_overlay_bgr(wrist_img, result["wrist"])
            wrist_overlay = label_frame(wrist_overlay, ["wrist cam"])
            panels.append(wrist_overlay)
        combined = np.hstack(panels)
        cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.png"), combined)

        per_frame_rows.append({
            "episode": ep_name, "frame": frame_idx, "traj_error": round(traj_error, 5),
            "state_share": round(result["state_share"], 4),
            "latent_share": round(result["latent_share"], 4),
        })

    # ---- Encode annotated video ----
    out_video = ep_out_dir / f"{ep_name}_attention.mp4"
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "%06d.png"),
           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out_video)]
    result_ff = subprocess.run(cmd, capture_output=True)
    if result_ff.returncode != 0:
        print(f"  ffmpeg error for {ep_name}: {result_ff.stderr.decode()[-500:]}")
    else:
        shutil.rmtree(frames_dir)
        print(f"  video saved: {out_video}")

    return per_frame_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path, help="Path to packaged_dataset (manifest.json + data/ + videos/)")
    p.add_argument("--max-episodes", type=int, default=None, help="Smoke test with a small number first, e.g. 1")
    p.add_argument("--out-dir", type=Path, default=Path("./outputs/dataset_analysis"))
    p.add_argument("--fps", type=int, default=15, help="Output video frame rate")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, config = load_model(args.checkpoint, device)

    manifest = json.loads((args.data_root / "manifest.json").read_text())
    episode_names = [e["name"] for e in manifest["episodes"]]
    if args.max_episodes is not None:
        episode_names = episode_names[:args.max_episodes]
    print(f"Analyzing {len(episode_names)} episode(s): {episode_names}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for ep_name in episode_names:
        print(f"Episode {ep_name}...")
        rows = analyze_episode(model, config, ep_name, args.data_root, args.out_dir, fps=args.fps)
        all_rows.extend(rows)

    per_frame_csv = args.out_dir / "per_frame_stats.csv"
    with open(per_frame_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else
                                 ["episode", "frame", "traj_error", "state_share", "latent_share"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Per-frame stats: {per_frame_csv}")

    # ---- Per-episode summary ----
    df_all = pd.DataFrame(all_rows)
    if not df_all.empty:
        summary = df_all.groupby("episode").agg(
            mean_traj_error=("traj_error", "mean"),
            max_traj_error=("traj_error", "max"),
            mean_state_share=("state_share", "mean"),
            mean_latent_share=("latent_share", "mean"),
            n_frames=("frame", "count"),
        ).reset_index()
        summary_csv = args.out_dir / "episode_summary.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"Episode summary: {summary_csv}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()