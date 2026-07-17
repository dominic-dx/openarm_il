import argparse
import json
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_DIR = SCRIPT_DIR.parent / "packaged_dataset"

def build_index_csv(local_dir: Path):
    manifest_path = local_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest["episodes"] if isinstance(manifest, dict) else manifest

    rows = []
    for e in episodes:
        rows.append({
            "root": e["source_root"],
            "episode_index": e["source_episode_index"],
            "partition": e["source_root"],
            "include": True,
        })
        rows.append({
            "root": e["source_root"],
            "episode_index": e["source_episode_index"],
            "partition": "combined",
            "include": True,
        })

    df = pd.DataFrame(rows)
    index_path = local_dir / "index.csv"
    df.to_csv(index_path, index=False)
    print(f"Generated index.csv at {index_path} with {len(df)} rows.")

def fetch_dataset(repo_id: str, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    manifest_path = local_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {local_dir}")
    build_index_csv(local_dir)
    print(f"Dataset ready at {local_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="dominicdx/master")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    args = parser.parse_args()
    fetch_dataset(args.repo_id, args.local_dir)