#!/usr/bin/env python3
"""Minimal ACT implementation: ResNet18 backbone(s) + Transformer encoder-decoder + CVAE latent.

Supports an optional second camera (wrist) via use_wrist_cam=True. When enabled,
a second ResNet18 backbone (not shared weights, separate instance) encodes the
wrist image, and its tokens are concatenated onto the transformer memory
alongside the chest-cam tokens, state, and latent -- exactly how ACT's original
paper handles multi-camera setups.
"""

import math

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops.misc import FrozenBatchNorm2d


def freeze_batchnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features)
            frozen.weight.data = child.weight.data.clone()
            frozen.bias.data = child.bias.data.clone()
            frozen.running_mean.data = child.running_mean.data.clone()
            frozen.running_var.data = child.running_var.data.clone()
            setattr(module, name, frozen)
        else:
            freeze_batchnorm(child)


class SinusoidalPosEmb2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        b, c, h, w = x.shape
        y_pos = torch.linspace(0, 2 * math.pi, h, device=x.device).unsqueeze(1).repeat(1, w)
        x_pos = torch.linspace(0, 2 * math.pi, w, device=x.device).unsqueeze(0).repeat(h, 1)
        d4 = self.dim // 4
        div = torch.exp(torch.arange(d4, device=x.device).float() * (-math.log(10000.0) / d4))
        pe = torch.zeros(self.dim, h, w, device=x.device)
        pe[0:d4] = torch.sin(y_pos.unsqueeze(0) * div.view(-1, 1, 1))
        pe[d4:2 * d4] = torch.cos(y_pos.unsqueeze(0) * div.view(-1, 1, 1))
        pe[2 * d4:3 * d4] = torch.sin(x_pos.unsqueeze(0) * div.view(-1, 1, 1))
        pe[3 * d4:] = torch.cos(x_pos.unsqueeze(0) * div.view(-1, 1, 1))
        return pe.unsqueeze(0).expand(b, -1, -1, -1)


def build_vision_backbone(pretrained_backbone, hidden_dim):
    weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
    backbone = models.resnet18(weights=weights)
    freeze_batchnorm(backbone)
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
    pos_embed = SinusoidalPosEmb2D(hidden_dim)
    return backbone, proj, pos_embed


class ACTModel(nn.Module):
    def __init__(self, state_dim=24, action_dim=24, chunk_size=100,
                 hidden_dim=512, n_heads=8, n_enc_layers=4, n_dec_layers=7,
                 latent_dim=32, pretrained_backbone=False, use_wrist_cam=False):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.use_wrist_cam = use_wrist_cam

        self.backbone, self.backbone_proj, self.img_pos_embed = build_vision_backbone(
            pretrained_backbone, hidden_dim)

        if use_wrist_cam:
            self.wrist_backbone, self.wrist_backbone_proj, self.wrist_pos_embed = build_vision_backbone(
                pretrained_backbone, hidden_dim)

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

    def _encode_image(self, image, backbone, proj, pos_embed):
        feat = backbone(image)             # (B, C, H', W')
        feat = proj(feat)                  # (B, D, H', W')
        pos = pos_embed(feat)              # (B, D, H', W')
        feat = feat + pos
        return feat.flatten(2).permute(0, 2, 1)  # (B, H'*W', D)

    def encode_image(self, image):
        return self._encode_image(image, self.backbone, self.backbone_proj, self.img_pos_embed)

    def encode_wrist_image(self, wrist_image):
        return self._encode_image(wrist_image, self.wrist_backbone, self.wrist_backbone_proj,
                                   self.wrist_pos_embed)

    def forward(self, image, state, action_chunk=None, is_pad=None, wrist_image=None):
        img_tokens = self.encode_image(image)               # (B, N, D)
        state_token = self.state_proj(state).unsqueeze(1)   # (B, 1, D)

        if self.use_wrist_cam:
            if wrist_image is None:
                raise ValueError("use_wrist_cam=True but no wrist_image was passed to forward()")
            wrist_tokens = self.encode_wrist_image(wrist_image)  # (B, N, D)
            vision_tokens = torch.cat([img_tokens, wrist_tokens], dim=1)
        else:
            vision_tokens = img_tokens

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
        memory = torch.cat([vision_tokens, state_token, latent_token], dim=1)

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