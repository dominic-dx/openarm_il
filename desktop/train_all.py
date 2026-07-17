import argparse
import json
import subprocess
from pathlib import Path

DATASET_ROOTS = ["dataset", "dataset_100", "dataset_quality"]

def get_episode_names(manifest_path: Path, source_root: str = None):
    manifest = json.loads(manifest_path.read_text())
    if source_root is None:
        return [e["name"] for e in manifest]
    return [e["name"] for e in manifest if e["source_root"] == source_root]

def run_training(data_root, output_dir, name_prefix, episode_filter, args):
    cmd = [
        "python", "train.py",
        "--data-root", data_root,
        "--output-dir", output_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--chunk-size", str(args.chunk_size),
        "--save-every", str(args.save_every),
        "--frame-stride", str(args.frame_stride),
        "--name-prefix", name_prefix,
    ]
    if episode_filter:
        cmd += ["--episode-filter", episode_filter]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../packaged_dataset")
    parser.add_argument("--checkpoints-root", default="../checkpoints")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=10)
    args = parser.parse_args()

    manifest_path = Path(args.data_root) / "manifest.json"

    for root in DATASET_ROOTS:
        out_dir = f"{args.checkpoints_root}/{root}_run"
        run_training(args.data_root, out_dir, root, root, args)

    combined_out = f"{args.checkpoints_root}/combined_run"
    run_training(args.data_root, combined_out, "combined", None, args)

if __name__ == "__main__":
    main()