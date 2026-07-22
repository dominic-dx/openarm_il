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


## rollout.py usage
After downloading, use the following command to rollout

python rollout.py --checkpoints ./checkpoints/modelX --n-action-steps 50


## DATASETS SO FAR:
So far, stored locally on the linux PC. Models are available on HF (may be private repo)
Use replay.py to see the model in physical space, otherwise just watch the videos.

### The datasets were recorded in slightly different environments. 
### The following were recorded on the green pallet, with ONE overhead camera
- "dataset_combined": 275 successful demonstrations. Combination of the following datasets:
- "dataset": ~90 successful demonstrations. Otmane and Dominic demonstrations 
- "dataset_quality": ~90 successful demonstrations. 
    - First half is a high-drop demonstration, second half 
    - Second half is "ideal" demonstrations
- "dataset_100": 100 successful demonstrations 

### The following were recorded on the white pallet with white background, with TWO cameras
- "dataset_clean": 100 successful demonstrations
    - First half is normal successes
    - Second half intentionally tries to guide the wrist camera to "look" for the target,
    then continue the pick and place task.

### The following were recorded in the RPL through lerobot
- "dataset_lerobot": 100 successful episodes recorded using Lerobot framework
    - NOTE: Not hosted online... probably floating in the huggingface cache folder in the RPL.