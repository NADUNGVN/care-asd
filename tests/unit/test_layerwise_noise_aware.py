from __future__ import annotations

import torch
from torch import Tensor, nn

from care_asd.models.layerwise_noise_aware import (
    LayerwiseNoiseAwareEncoder,
    finite_adapter_update,
)


class _ZeroPosition(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        return torch.zeros_like(values)


class _FrozenLayer(nn.Module):
    def forward(
        self,
        values: Tensor,
        *,
        self_attn_padding_mask: Tensor | None,
        need_weights: bool,
        pos_bias: Tensor | None,
    ) -> tuple[Tensor, None, Tensor | None]:
        del self_attn_padding_mask, need_weights
        return values + 0.1, None, pos_bias


class _FakeBEATs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([_FrozenLayer(), _FrozenLayer()])
        self.encoder.pos_conv = _ZeroPosition()
        self.encoder.layer_norm = nn.Identity()
        self.encoder.layer_norm_first = False
        self.encoder.layerdrop = 0.0
        self.encoder.dropout = 0.0


def _model() -> LayerwiseNoiseAwareEncoder:
    return LayerwiseNoiseAwareEncoder(
        beats_model=_FakeBEATs(),
        frequency_patches=2,
        embedding_dim=4,
        hidden_dim=4,
        attention_heads=1,
        dropout=0.0,
        insertion_layers=(1, 2),
    )


def test_zero_initialized_layerwise_adapter_preserves_frozen_path() -> None:
    model = _model().eval()
    target = torch.randn(3, 2, 2, 4)
    reference = torch.randn_like(target)
    result = model(target, reference)
    torch.testing.assert_close(result, target + 0.2)


def test_configured_layerdrop_is_inactive_on_the_frozen_eval_path() -> None:
    beats = _FakeBEATs()
    beats.encoder.layerdrop = 0.05
    model = LayerwiseNoiseAwareEncoder(
        beats_model=beats,
        frequency_patches=2,
        embedding_dim=4,
        hidden_dim=4,
        attention_heads=1,
        dropout=0.0,
        insertion_layers=(1, 2),
    ).train()
    target = torch.randn(2, 2, 2, 4)
    reference = torch.randn_like(target)
    result = model(target, reference)
    torch.testing.assert_close(result, target + 0.2)
    assert model.beats_model.training is False
    assert all(layer.training is False for layer in model.beats_model.encoder.layers)


def test_layerwise_optimizer_updates_only_adapter_state() -> None:
    model = _model().train()
    before = model.clone_adapter_state_dict()
    target = torch.randn(2, 2, 2, 4)
    reference = torch.randn_like(target)
    loss = model(target, reference).square().mean()
    loss.backward()
    optimizer = torch.optim.SGD(model.adapters.parameters(), lr=0.1)
    optimizer.step()
    updated, norm = finite_adapter_update(before, model.adapter_state_dict())
    assert updated
    assert norm > 0.0
    assert all(not parameter.requires_grad for parameter in model.beats_model.parameters())


def test_layerwise_grid_contract_is_strict() -> None:
    model = _model()
    good = torch.zeros(1, 2, 2, 4)
    bad = torch.zeros(1, 2, 1, 4)
    try:
        model(good, bad)
    except ValueError as error:
        assert "contract" in str(error) or "equal shapes" in str(error)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("invalid frequency grid was accepted")
