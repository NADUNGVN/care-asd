"""Capacity-matched near-primary, reliability-gated CARE residual autoencoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GatedNearResidualAutoencoder(nn.Module):
    """B02: two 640-to-64 branches equal the B00 640-to-128 first layer."""

    def __init__(self, input_dim: int = 640) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.near_branch = _block(input_dim, 64)
        self.residual_branch = _block(input_dim, 64)
        self.encoder_tail = nn.Sequential(
            _block(128, 128), _block(128, 128), _block(128, 128), _block(128, 8)
        )
        self.decoder = nn.Sequential(
            _block(8, 128),
            _block(128, 128),
            _block(128, 128),
            _block(128, 128),
            nn.Linear(128, input_dim),
        )

    def forward(self, near: Tensor, residual: Tensor, reliability: Tensor) -> tuple[Tensor, Tensor]:
        """Reconstruct untouched near features; reliability gates only the auxiliary branch."""
        near_flat = near.reshape(-1, self.input_dim)
        residual_flat = residual.reshape(-1, self.input_dim)
        gate = reliability.reshape(-1, 1).clamp(0.0, 1.0)
        combined = torch.cat(
            (self.near_branch(near_flat), gate * self.residual_branch(residual_flat)), dim=1
        )
        latent = self.encoder_tail(combined)
        return self.decoder(latent), latent


def _block(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim, momentum=0.01, eps=1.0e-3),
        nn.ReLU(),
    )
