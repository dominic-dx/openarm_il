#!/usr/bin/env python3
"""Minimal ACT implementation: ResNet18 backbone + Transformer encoder-decoder + CVAE latent."""

import torch
import torch.nn as nn
import torchvision.models as models


class ACTModel(nn.Module):
    def __init__(self, state_dim=24, action_dim=24, chunk_size=100,
                 hidden_dim=512, n_heads=8, n_enc_layers=4, n_dec_layers=7,
                 latent_dim=32, pretrained_backbone=False):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.backbone_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        self.cvae_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, n_heads, batch_first=True), n_enc_layers
        )
        self.latent_head = nn.Linear(hidden_dim, latent_dim * 2)  # mu, logvar

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, n_heads, batch_first=True), n_dec_layers
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.query_embed = nn.Parameter(torch.randn(chunk_size, hidden_dim))

    def encode_image(self, image):
        feat = self.backbone(image)                       # (B, C, H', W')
        feat = self.backbone_proj(feat)
        B, C, H, W = feat.shape
        return feat.flatten(2).permute(0, 2, 1)            # (B, H'*W', C)

    def forward(self, image, state, action_chunk=None, is_pad=None):
        img_tokens = self.encode_image(image)               # (B, N, D)
        state_token = self.state_proj(state).unsqueeze(1)   # (B, 1, D)

        if action_chunk is not None:
            action_tokens = self.action_proj(action_chunk)  # (B, T, D)
            cvae_input = torch.cat([state_token, action_tokens], dim=1)
            enc_out = self.cvae_encoder(cvae_input)
            latent_params = self.latent_head(enc_out[:, 0])
            mu, logvar = latent_params.chunk(2, dim=-1)
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
        else:
            B = image.shape[0]
            z = torch.zeros(B, self.latent_dim, device=image.device)
            mu = logvar = None

        latent_token = self.latent_proj(z).unsqueeze(1)     # (B, 1, D)
        memory = torch.cat([img_tokens, state_token, latent_token], dim=1)

        B = image.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(queries, memory)
        pred_actions = self.action_head(decoded)             # (B, T, action_dim)

        return pred_actions, mu, logvar


def act_loss(pred_actions, target_actions, is_pad, mu, logvar, kl_weight=10.0):
    mask = (~is_pad).unsqueeze(-1).float()
    l1 = (pred_actions - target_actions).abs() * mask
    recon_loss = l1.sum() / mask.sum().clamp(min=1)

    if mu is not None:
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    else:
        kl = torch.tensor(0.0, device=pred_actions.device)

    return recon_loss + kl_weight * kl, recon_loss, kl