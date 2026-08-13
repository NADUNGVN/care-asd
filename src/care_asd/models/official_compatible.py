"""Architecture-level reproduction of the pinned DCASE 2026 AE baseline."""

from __future__ import annotations

from torch import Tensor, nn


class OfficialCompatibleAutoencoder(nn.Module):
    """The official 640-to-128 (four layers)-to-8 AE BatchNorm contract."""

    def __init__(self, input_dim: int = 640) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1.0e-3),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1.0e-3),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1.0e-3),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1.0e-3),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.BatchNorm1d(8, momentum=0.01, eps=1.0e-3),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1.0e-3),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1.0e-3),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1.0e-3),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1.0e-3),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Return reconstruction and 8-dimensional latent code exactly as official."""
        flattened = values.reshape(-1, self.input_dim)
        latent = self.encoder(flattened)
        return self.decoder(latent), latent
