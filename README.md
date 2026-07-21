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

## Fetch models for rollout
**It's a private repo, need to log in with my token**
hf download aiet2/dataset-combined-run ./checkpoints/model1
hf download aiet2/dataset-100-run ./checkpoints/model2
hf download aiet2/openarm-il-dataset-run ./checkpoints/model3
hf download aiet2/dataset-quality-run ./checkpoints/model4
hf download aiet2/dataset-clean-run ./checkpoints/model5
**The structure of model5 has 6 epochs inside it: model5/epoch{2-7}**