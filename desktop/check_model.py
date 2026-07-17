import torch, json
from pathlib import Path
from model import ACTModel

ckpt_dir = Path("./checkpoints/smoke_test")
config = json.loads((ckpt_dir / "config.json").read_text())
model = ACTModel(state_dim=config["state_dim"], action_dim=config["action_dim"], chunk_size=config["chunk_size"])
model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))
model.eval()

dummy_image = torch.zeros(1, 3, 480, 848)
dummy_state = torch.zeros(1, 24)
with torch.no_grad():
    pred_actions, _, _ = model(dummy_image, dummy_state)

print(pred_actions.shape)   # should be (1, chunk_size, 24)
print(pred_actions[0, 0])   # first predicted action vector, should be finite numbers, not NaN/inf

from dataset_video import PackagedEpisodeDataset

dataset = PackagedEpisodeDataset("./packaged_dataset", chunk_size=config["chunk_size"], arm_role="fr", limit_episodes=3)
sample = dataset[0]

state_mean = torch.tensor(config["norm_stats"]["state_mean"])
state_std = torch.tensor(config["norm_stats"]["state_std"])
norm_state = ((sample["state"] - state_mean) / state_std).unsqueeze(0)
image = sample["image"].unsqueeze(0)

with torch.no_grad():
    pred_actions, _, _ = model(image, norm_state)

print("predicted first action:", pred_actions[0, 0, 0::3])
print("ground truth first action:", sample["action_chunk"][0, 0::3])