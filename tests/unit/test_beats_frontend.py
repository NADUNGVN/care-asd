from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from care_asd.models.beats_frontend import OfficialBEATsFrontend


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_norm_first = False
        self.layer_norm = _Offset(100.0)

    def forward(
        self, values: torch.Tensor, padding_mask: object = None, layer: int | None = None
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, None]]]:
        del padding_mask
        time_major = values.transpose(0, 1)
        results = [(time_major, None)]
        depth = 2 if layer is None else layer + 1
        for _ in range(depth):
            time_major = time_major + 1.0
            results.append((time_major, None))
        return time_major.transpose(0, 1), results


class _Offset(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.value


class _FakeBEATs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(encoder_layers=2)
        self.patch_embedding = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        nn.init.ones_(self.patch_embedding.weight)
        self.layer_norm = nn.Identity()
        self.post_extract_proj = None
        self.dropout_input = nn.Identity()
        self.encoder = _FakeEncoder()

    def preprocess(self, source: torch.Tensor) -> torch.Tensor:
        return source.reshape(len(source), 2, 2)


def _frontend() -> OfficialBEATsFrontend:
    frontend = OfficialBEATsFrontend.__new__(OfficialBEATsFrontend)
    frontend._torch = torch
    frontend._model = _FakeBEATs()
    frontend.device = "cpu"
    frontend.frequency_patches = 2
    frontend.mixed_precision = False
    return frontend


def test_extract_encoder_taps_preserves_declared_depths() -> None:
    result = _frontend().extract_encoder_taps(
        np.arange(8, dtype=np.float32).reshape(2, 4), taps=(0, 1, 2)
    )
    assert set(result) == {0, 1, 2}
    assert result[0].shape == (2, 2, 2, 3)
    np.testing.assert_allclose(result[1], result[0] + 1.0)
    np.testing.assert_allclose(result[2], result[0] + 2.0)


def test_final_encoder_tap_applies_the_official_pre_norm_epilogue() -> None:
    frontend = _frontend()
    frontend._model.encoder.layer_norm_first = True
    result = frontend.extract_encoder_taps(
        np.arange(4, dtype=np.float32).reshape(1, 4), taps=(0, 2)
    )
    np.testing.assert_allclose(result[2], result[0] + 102.0)


@pytest.mark.parametrize("taps", [(), (1, 0), (0, 0), (0, 3)])
def test_extract_encoder_taps_rejects_invalid_contract(taps: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="taps"):
        _frontend().extract_encoder_taps(np.ones((1, 4), dtype=np.float32), taps=taps)
