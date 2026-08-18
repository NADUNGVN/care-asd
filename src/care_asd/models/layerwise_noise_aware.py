"""Capacity-matched layerwise noise-aware BEATs adapter used by FP-NAA v8."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from care_asd.models.fp_naa_adapter import BandwiseReferenceAdapter


class LayerwiseNoiseAwareEncoder(nn.Module):
    """Run frozen BEATs blocks with a trainable near-only adapter after each block.

    Inputs are BEATs tap-0 grids with shape ``[batch, time, frequency, embedding]``. The
    upstream patch frontend and all original BEATs parameters remain frozen. Only the adapters
    are included in :meth:`adapter_state_dict`, so checkpoints never duplicate the upstream
    Iteration-3 weights.
    """

    def __init__(
        self,
        *,
        beats_model: nn.Module,
        frequency_patches: int = 8,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        attention_heads: int = 8,
        dropout: float = 0.1,
        insertion_layers: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if frequency_patches < 1 or embedding_dim < 1:
            raise ValueError("frequency_patches and embedding_dim must be positive")
        encoder = getattr(beats_model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if encoder is None or layers is None or len(layers) < 1:
            raise ValueError("beats_model must expose a non-empty encoder.layers sequence")
        layer_count = len(layers)
        chosen = insertion_layers or tuple(range(1, layer_count + 1))
        if not chosen or tuple(sorted(set(chosen))) != chosen:
            raise ValueError("insertion_layers must be a sorted tuple of unique depths")
        if chosen[0] < 1 or chosen[-1] > layer_count:
            raise ValueError(f"insertion_layers must be in [1, {layer_count}]")

        self.beats_model = beats_model
        self.frequency_patches = frequency_patches
        self.embedding_dim = embedding_dim
        self.insertion_layers = chosen
        self.adapters = nn.ModuleDict(
            {
                str(depth): BandwiseReferenceAdapter(
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    conditioning_mode="target_conditioned",
                )
                for depth in chosen
            }
        )
        for parameter in self.beats_model.parameters():
            parameter.requires_grad_(False)
        self.beats_model.eval()

    def train(self, mode: bool = True) -> LayerwiseNoiseAwareEncoder:
        """Train adapters while keeping every upstream BEATs module in evaluation mode."""
        super().train(mode)
        self.beats_model.eval()
        for adapter in self.adapters.values():
            adapter.train(mode)
        return self

    def forward(self, target_tap0: Tensor, reference_tap0: Tensor) -> Tensor:
        """Return the final target grid after all frozen blocks and NA residuals."""
        self._validate_grid(target_tap0, "target_tap0")
        self._validate_grid(reference_tap0, "reference_tap0")
        if target_tap0.shape != reference_tap0.shape:
            raise ValueError("target and reference tap-0 grids must have equal shapes")

        batch, time, bands, dimension = target_tap0.shape
        target = target_tap0.reshape(batch, time * bands, dimension)
        reference = reference_tap0.reshape(batch, time * bands, dimension)
        target = self._encoder_input(target)
        with torch.no_grad():
            reference = self._encoder_input(reference)
        target = target.transpose(0, 1)
        reference = reference.transpose(0, 1)
        target_bias: Tensor | None = None
        reference_bias: Tensor | None = None
        encoder = self.beats_model.encoder

        for depth, layer in enumerate(encoder.layers, start=1):
            with torch.no_grad():
                reference, _, reference_bias = layer(
                    reference,
                    self_attn_padding_mask=None,
                    need_weights=False,
                    pos_bias=reference_bias,
                )
            target, _, target_bias = layer(
                target,
                self_attn_padding_mask=None,
                need_weights=False,
                pos_bias=target_bias,
            )
            adapter_key = str(depth)
            if adapter_key in self.adapters:
                adapter = self.adapters[adapter_key]
                target_grid = target.transpose(0, 1).reshape(batch, time, bands, dimension)
                reference_grid = reference.transpose(0, 1).reshape(batch, time, bands, dimension)
                target_grid = adapter(target_grid, reference_grid)
                target = target_grid.reshape(batch, time * bands, dimension).transpose(0, 1)

        target = target.transpose(0, 1)
        if bool(getattr(encoder, "layer_norm_first", False)):
            target = encoder.layer_norm(target)
        return target.reshape(batch, time, bands, dimension)

    def adapter_state_dict(self) -> dict[str, Tensor]:
        """Return a detached CPU copy containing trainable V8 state only."""
        return {
            name: value.detach().cpu().clone() for name, value in self.adapters.state_dict().items()
        }

    def load_adapter_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore a strict trainable-only checkpoint."""
        self.adapters.load_state_dict(dict(state), strict=True)

    def clone_adapter_state_dict(self) -> dict[str, Tensor]:
        """Alias emphasizing that the returned branch point is immutable."""
        return copy.deepcopy(self.adapter_state_dict())

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.adapters.parameters())

    def _encoder_input(self, values: Tensor) -> Tensor:
        encoder = self.beats_model.encoder
        positional = encoder.pos_conv(values.transpose(1, 2)).transpose(1, 2)
        output = values + positional
        if not bool(getattr(encoder, "layer_norm_first", False)):
            output = encoder.layer_norm(output)
        return functional.dropout(output, p=float(encoder.dropout), training=False)

    def _validate_grid(self, values: Tensor, name: str) -> None:
        if values.ndim != 4:
            raise ValueError(f"{name} must have shape [batch, time, frequency, embedding]")
        if values.shape[2] != self.frequency_patches or values.shape[3] != self.embedding_dim:
            raise ValueError(f"{name} violates the frozen BEATs grid contract")
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must be finite")


def finite_adapter_update(
    before: Mapping[str, Tensor], after: Mapping[str, Tensor]
) -> tuple[bool, float]:
    """Validate and quantify a real finite trainable-state update."""
    if set(before) != set(after):
        raise ValueError("adapter state dictionaries do not have identical keys")
    squared = 0.0
    for name in sorted(before):
        left = before[name].float()
        right = after[name].float()
        if left.shape != right.shape or not torch.isfinite(right).all():
            raise ValueError(f"Invalid adapter update tensor: {name}")
        squared += float((right - left).square().sum().item())
    norm = squared**0.5
    return norm > 0.0, norm
