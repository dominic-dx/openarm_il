import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def fetch_dataset(repo_id: str, local_dir: str):
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    manifest_path = local_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {local_dir} — download may have failed or repo isn't in our packaged format.")
    print(f"Dataset downloaded to {local_dir}, manifest found with expected structure.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="dominicdx/master")
    parser.add_argument("--local-dir", default="../packaged_dataset")
    args = parser.parse_args()
    fetch_dataset(args.repo_id, args.local_dir)