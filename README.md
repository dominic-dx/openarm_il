## Setup
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # or cu126 for GPU

## Fetch dataset
python desktop/fetch_dataset.py --repo-id dominicdx/master --local-dir ../packaged_dataset

## Train all variants
python desktop/train_all.py --epochs 8

## Push a trained checkpoint
python desktop/push_checkpoints.py --local-dir ../checkpoints/combined_run --repo-id your-username/repo-name