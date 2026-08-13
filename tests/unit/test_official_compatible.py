"""Architecture contract for the internal official-alignment model."""

from __future__ import annotations

import pytest

from care_asd.models.official_compatible import OfficialCompatibleAutoencoder

torch = pytest.importorskip("torch")


def test_official_compatible_autoencoder_matches_vector_and_latent_dimensions() -> None:
    model = OfficialCompatibleAutoencoder()
    reconstruction, latent = model(torch.randn(3, 640))

    assert reconstruction.shape == (3, 640)
    assert latent.shape == (3, 8)
