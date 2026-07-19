## Setup
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 for GPU or /cpu for CPU

## Fetch dataset
python desktop/fetch_dataset.py --repo-id dominicdx/master --local-dir ../packaged_dataset
python desktop/fetch_dataset.py --repo-id dominicdx/clean --local-dir ../packaged_clean

## Train all variants
python desktop/train_all.py --epochs 8

## Push a trained checkpoint
python desktop/push_checkpoints.py --local-dir ../checkpoints/combined_run --repo-id your-username/repo-name
