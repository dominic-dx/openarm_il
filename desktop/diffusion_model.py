#!/usr/bin/env python3
"""
Minimal Diffusion Policy (DDPM-style) for the same 24-dim state / 24-dim
action-chunk task as ACTModel in model.py. Uses the SAME ResNet18 vision
backbone + 2D sinusoidal positional embedding pattern as your existing
model.py, so it's a drop-in alternative you can A/B test against ACT.

Architecture:
  - ResNet18 (optionally ImageNet-pretrained) encodes the chest image into
    a set of spatial tokens, pooled into a single global vision embedding.
  - State is projected into the same embedding space.
  - A small MLP-based noise-prediction network conditions on
    [vision_embed, state_embed, diffusion_timestep_embed] and predicts the
    noise added to a noised action chunk.
  - Standard DDPM forward/reverse process, ~50-100 denoising steps at
    inference (adjustable).

This is intentionally minimal -- no U-Net, no transformer denoiser -- to
keep it fast enough to iterate on CPU/laptop-GPU. Swap in a bigger
denoiser later if this proves promising.
"""

import math
import torch
import torch.nn as nn
import torchvision.models as models


def build_vision_backbone(pretrained_backbone: bool, hidden_dim: int):
    weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
    backbone = models.resnet18(weights=weights)
    backbone = nn.Sequential(*list(backbone.children())[:-1])  # drop avgpool+fc, keep up to last conv... actually resnet18's [:-1] keeps avgpool, drops fc
    proj = nn.Linear(512, hidden_dim)
    return backbone, proj


class SinusoidalTimestepEmb(nn.Module):
    """Standard DDPM sinusoidal embedding for the diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: (B,) long tensor of timesteps
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class NoisePredictor(nn.Module):
    """Predicts noise epsilon given [vision, state, timestep] conditioning
    and the current noised action chunk. Simple conditioned MLP over the
    flattened chunk -- fast to train, easy to debug."""
    def __init__(self, chunk_size, action_dim, hidden_dim=512, cond_dim=512):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        flat_dim = chunk_size * action_dim

        self.time_embed = SinusoidalTimestepEmb(cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim * 3, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
        )
        self.action_in = nn.Linear(flat_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def forward(self, noisy_actions, t, vision_embed, state_embed):
        B = noisy_actions.shape[0]
        t_embed = self.time_embed(t)  # (B, cond_dim)
        cond = torch.cat([vision_embed, state_embed, t_embed], dim=-1)
        cond = self.cond_proj(cond)  # (B, hidden_dim)

        x = self.action_in(noisy_actions.reshape(B, -1))  # (B, hidden_dim)
        h = torch.cat([x, cond], dim=-1)
        eps_pred = self.net(h)
        return eps_pred.reshape(B, self.chunk_size, self.action_dim)


class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim=24, action_dim=24, chunk_size=50, hidden_dim=512,
                 pretrained_backbone=False, use_wrist_cam=False, n_diffusion_steps=100):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.use_wrist_cam = use_wrist_cam
        self.n_diffusion_steps = n_diffusion_steps

        self.backbone, self.backbone_proj = build_vision_backbone(pretrained_backbone, hidden_dim)
        if use_wrist_cam:
            self.wrist_backbone, self.wrist_backbone_proj = build_vision_backbone(pretrained_backbone, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        self.noise_predictor = NoisePredictor(chunk_size, action_dim, hidden_dim, cond_dim=hidden_dim)

        betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def encode_vision(self, image):
        feat = self.backbone(image).flatten(1)  # (B, 512)
        return self.backbone_proj(feat)  # (B, hidden_dim)

    def encode_wrist(self, wrist_image):
        feat = self.wrist_backbone(wrist_image).flatten(1)
        return self.wrist_backbone_proj(feat)

    def forward_train(self, image, state, action_chunk, wrist_image=None):
        """Training step: sample random t, noise the action chunk, predict noise, return MSE loss."""
        B = image.shape[0]
        device = image.device

        vision_embed = self.encode_vision(image)
        if self.use_wrist_cam:
            if wrist_image is None:
                raise ValueError("use_wrist_cam=True but no wrist_image passed")
            vision_embed = vision_embed + self.encode_wrist(wrist_image)  # simple additive fusion
        state_embed = self.state_proj(state)

        t = torch.randint(0, self.n_diffusion_steps, (B,), device=device)
        noise = torch.randn_like(action_chunk)
        sqrt_acp = self.alphas_cumprod[t].sqrt().view(B, 1, 1)
        sqrt_one_minus_acp = (1 - self.alphas_cumprod[t]).sqrt().view(B, 1, 1)
        noisy_actions = sqrt_acp * action_chunk + sqrt_one_minus_acp * noise

        eps_pred = self.noise_predictor(noisy_actions, t, vision_embed, state_embed)
        loss = nn.functional.mse_loss(eps_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, image, state, wrist_image=None, n_steps=None):
        """Inference: iterative DDPM denoising from pure noise to a predicted action chunk."""
        n_steps = n_steps or self.n_diffusion_steps
        B = image.shape[0]
        device = image.device

        vision_embed = self.encode_vision(image)
        if self.use_wrist_cam:
            vision_embed = vision_embed + self.encode_wrist(wrist_image)
        state_embed = self.state_proj(state)

        x = torch.randn(B, self.chunk_size, self.action_dim, device=device)
        for t_step in reversed(range(n_steps)):
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            eps_pred = self.noise_predictor(x, t, vision_embed, state_embed)

            alpha_t = 1.0 - self.betas[t_step]
            acp_t = self.alphas_cumprod[t_step]
            acp_prev = self.alphas_cumprod[t_step - 1] if t_step > 0 else torch.tensor(1.0, device=device)

            x0_pred = (x - (1 - acp_t).sqrt() * eps_pred) / acp_t.sqrt()
            x0_pred = x0_pred.clamp(-5, 5)  # simple stability clamp, actions are normalized

            if t_step > 0:
                mean = acp_prev.sqrt() * self.betas[t_step] / (1 - acp_t) * x0_pred + \
                       alpha_t.sqrt() * (1 - acp_prev) / (1 - acp_t) * x
                sigma = ((1 - acp_prev) / (1 - acp_t) * self.betas[t_step]).sqrt()
                x = mean + sigma * torch.randn_like(x)
            else:
                x = x0_pred
        return x


if __name__ == "__main__":
    model = DiffusionPolicy(chunk_size=50, pretrained_backbone=False)
    dummy_image = torch.zeros(2, 3, 480, 640)
    dummy_state = torch.zeros(2, 24)
    dummy_actions = torch.zeros(2, 50, 24)
    loss = model.forward_train(dummy_image, dummy_state, dummy_actions)
    print(f"Training loss (should be finite scalar): {loss.item():.4f}")

    sampled = model.sample(dummy_image, dummy_state, n_steps=10)
    print(f"Sampled action chunk shape (should be (2, 50, 24)): {sampled.shape}")