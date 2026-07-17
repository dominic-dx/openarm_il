import argparse
import json
import subprocess
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train.py"
DATASET_ROOTS = ["dataset", "dataset_100", "dataset_quality"]

def run_training(data_root, output_dir, name_prefix, partition, index_csv, args):
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--data-root", data_root,
        "--output-dir", output_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--chunk-size", str(args.chunk_size),
        "--save-every", str(args.save_every),
        "--frame-stride", str(args.frame_stride),
        "--name-prefix", name_prefix,
        "--index-csv", index_csv,
        "--partition", partition,
    ]
    if args.limit_episodes:
        cmd += ["--limit-episodes", str(args.limit_episodes)]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(SCRIPT_DIR.parent / "packaged_dataset"))
    parser.add_argument("--checkpoints-root", default=str(SCRIPT_DIR.parent / "checkpoints"))
    parser.add_argument("--index-csv", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    if args.index_csv is None:
        args.index_csv = str(Path(args.data_root) / "index.csv")

    manifest_path = Path(args.data_root) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}. Run fetch_dataset.py first.")
    if not Path(args.index_csv).exists():
        raise FileNotFoundError(f"index.csv not found at {args.index_csv}. Run fetch_dataset.py first.")

    for root in DATASET_ROOTS:
        out_dir = f"{args.checkpoints_root}/{root}_run"
        run_training(args.data_root, out_dir, root, root, args.index_csv, args)

    combined_out = f"{args.checkpoints_root}/combined_run"
    run_training(args.data_root, combined_out, "combined", "combined", args.index_csv, args)

if __name__ == "__main__":
    main()