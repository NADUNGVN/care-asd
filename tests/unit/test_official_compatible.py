"""Architecture contract for the internal official-alignment model."""

from __future__ import annotations

import pytest

from care_asd.models.gated_fusion import GatedNearResidualAutoencoder
from care_asd.models.official_compatible import OfficialCompatibleAutoencoder

torch = pytest.importorskip("torch")


def test_official_compatible_autoencoder_matches_vector_and_latent_dimensions() -> None:
    model = OfficialCompatibleAutoencoder()
    reconstruction, latent = model(torch.randn(3, 640))

    assert reconstruction.shape == (3, 640)
    assert latent.shape == (3, 8)


def test_gated_fusion_matches_b00_parameter_count_and_near_output_shape() -> None:
    baseline = OfficialCompatibleAutoencoder()
    fusion = GatedNearResidualAutoencoder()
    output, latent = fusion(torch.randn(3, 640), torch.randn(3, 640), torch.ones(3))
    assert output.shape == (3, 640)
    assert latent.shape == (3, 8)
    assert sum(item.numel() for item in fusion.parameters()) == sum(
        item.numel() for item in baseline.parameters()
    )
