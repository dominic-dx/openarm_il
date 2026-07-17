import argparse
from pathlib import Path
from huggingface_hub import HfApi

def push_checkpoint(local_dir: str, repo_id: str, private: bool = True):
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Pushed {local_dir} to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", required=True, help="e.g. ../checkpoints/combined_run")
    parser.add_argument("--repo-id", required=True, help="e.g. dominicdx/openarm-combined")
    parser.add_argument("--private", action="store_true", default=True)
    args = parser.parse_args()
    push_checkpoint(args.local_dir, args.repo_id, args.private)