"""Compact GPU autoencoder used by the frozen Phase 5 MVP ablations."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn


class DepthwiseSeparableBlock(nn.Module):
    """Frequency-aware block; temporal resolution is preserved throughout."""

    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=(3, 3),
                stride=(stride, 1),
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm2d(input_channels),
            nn.SiLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return cast(Tensor, self.layers(values))


class LightweightNearAutoencoder(nn.Module):
    """Reconstruct near log-Mel from a configurable multi-view input tensor."""

    def __init__(self, input_channels: int, embedding_dim: int = 64) -> None:
        super().__init__()
        if input_channels < 1 or embedding_dim < 1:
            raise ValueError("input_channels and embedding_dim must be positive")
        self.encoder = nn.Sequential(
            DepthwiseSeparableBlock(input_channels, 32, stride=2),
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, 96, stride=2),
        )
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, embedding_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 64, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, values: Tensor) -> Tensor:
        encoded = cast(Tensor, self.encoder(values))
        reconstructed = cast(Tensor, self.decoder(encoded))
        return reconstructed[..., : values.shape[-2], : values.shape[-1]]

    def encode(self, values: Tensor) -> Tensor:
        """Return the compact representation for later extension/scoring work."""
        return cast(Tensor, self.embedding(self.encoder(values)))


def approximate_parameter_count(model: nn.Module) -> int:
    """Return trainable parameter count for the model card."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
