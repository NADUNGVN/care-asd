"""Capacity-matched reference-conditioned adapter for FP-NAA C1/C2/C3."""

from __future__ import annotations

import math
from typing import Literal, cast

import torch
from torch import Tensor, nn

ConditioningMode = Literal["target_conditioned", "reference_only_equivariant"]
ReferenceSafetyMode = Literal["none", "rdp_salient_contraction"]


class BandwiseReferenceAdapter(nn.Module):
    """Denoise frozen token grids with band-aligned cross-attention.

    Inputs and outputs use ``[batch, time, frequency_patch, embedding]``. Each frequency patch is
    processed independently with shared weights, preventing attention from silently mixing BEATs
    sub-bands before the common BEAM backend. The zero-initialized output projection makes the
    initial adapter an exact identity map.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        attention_heads: int = 8,
        dropout: float = 0.1,
        conditioning_mode: ConditioningMode = "target_conditioned",
        reference_safety_mode: ReferenceSafetyMode = "none",
        reference_safety_fraction: float = 0.20,
        maximum_reference_contraction: float = 1.0,
    ) -> None:
        super().__init__()
        if min(embedding_dim, hidden_dim, attention_heads) < 1:
            raise ValueError("embedding_dim, hidden_dim, and attention_heads must be positive")
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if conditioning_mode not in {"target_conditioned", "reference_only_equivariant"}:
            raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode}")
        if reference_safety_mode not in {"none", "rdp_salient_contraction"}:
            raise ValueError(f"Unsupported reference_safety_mode: {reference_safety_mode}")
        if not 0.0 < reference_safety_fraction <= 1.0:
            raise ValueError("reference_safety_fraction must be in (0, 1]")
        if not 0.0 <= maximum_reference_contraction <= 1.0:
            raise ValueError("maximum_reference_contraction must be in [0, 1]")
        self.embedding_dim = embedding_dim
        self.conditioning_mode = conditioning_mode
        self.reference_safety_mode = reference_safety_mode
        self.reference_safety_fraction = reference_safety_fraction
        self.maximum_reference_contraction = maximum_reference_contraction
        self.target_norm = nn.LayerNorm(embedding_dim)
        self.reference_norm = nn.LayerNorm(embedding_dim)
        self.target_projection = nn.Linear(embedding_dim, hidden_dim)
        self.reference_projection = nn.Linear(embedding_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        output = self.fusion[-1]
        if not isinstance(output, nn.Linear):
            raise AssertionError("FP-NAA output projection must be linear")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, target: Tensor, reference: Tensor) -> Tensor:
        """Return a target-shaped adapted grid conditioned on the reference grid."""
        if target.ndim != 4 or reference.ndim != 4:
            raise ValueError("target and reference must have shape [batch, time, band, embedding]")
        if target.shape != reference.shape:
            raise ValueError("target and reference token grids must have equal shapes")
        if target.shape[-1] != self.embedding_dim:
            raise ValueError("token embedding dimension does not match the adapter")
        batch, time, bands, dimension = target.shape
        target_rows = target.permute(0, 2, 1, 3).reshape(batch * bands, time, dimension)
        reference_rows = reference.permute(0, 2, 1, 3).reshape(batch * bands, time, dimension)
        query_source = (
            reference_rows
            if self.conditioning_mode == "reference_only_equivariant"
            else target_rows
        )
        query = self.target_projection(self.target_norm(query_source))
        key_value = self.reference_projection(self.reference_norm(reference_rows))
        attended, _ = self.cross_attention(
            query,
            key_value,
            key_value,
            need_weights=False,
        )
        correction = self.fusion(torch.cat((query, attended), dim=-1))
        if self.reference_safety_mode == "rdp_salient_contraction":
            correction = rdp_salient_contraction_projection(
                correction=correction.reshape(batch, bands, time, dimension)
                .permute(0, 2, 1, 3)
                .contiguous(),
                target=target,
                reference=reference,
                protected_fraction=self.reference_safety_fraction,
                maximum_contraction=self.maximum_reference_contraction,
            ).permute(0, 2, 1, 3).reshape(batch * bands, time, dimension)
        adapted = target_rows + correction
        return cast(
            Tensor,
            adapted.reshape(batch, bands, time, dimension).permute(0, 2, 1, 3).contiguous(),
        )


def rdp_salient_contraction_projection(
    *,
    correction: Tensor,
    target: Tensor,
    reference: Tensor,
    protected_fraction: float,
    maximum_contraction: float,
    eps: float = 1.0e-8,
) -> Tensor:
    """Limit correction toward the far channel on RDP-salient temporal rows.

    The near-minus-far discrepancy is treated as evidence unavailable from the reference
    microphone. On the temporal rows with the largest discrepancy, the projected correction
    cannot reduce that evidence by more than ``maximum_contraction``. The operation has no
    trainable parameters and is the identity whenever the raw correction is already safe.
    """
    if correction.shape != target.shape or correction.shape != reference.shape:
        raise ValueError("correction, target, and reference must have identical shapes")
    if correction.ndim != 4 or min(correction.shape) < 1:
        raise ValueError("projection tensors must have shape [batch, time, band, embedding]")
    if not 0.0 < protected_fraction <= 1.0:
        raise ValueError("protected_fraction must be in (0, 1]")
    if not 0.0 <= maximum_contraction <= 1.0:
        raise ValueError("maximum_contraction must be in [0, 1]")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    work_dtype = torch.float32 if correction.dtype in {torch.float16, torch.bfloat16} else correction.dtype
    discrepancy = (target - reference).to(dtype=work_dtype)
    projected = correction.to(dtype=work_dtype)
    discrepancy_norm_sq = discrepancy.square().sum(dim=(2, 3), keepdim=True)
    correction_dot = (projected * discrepancy).sum(dim=(2, 3), keepdim=True)
    minimum_dot = -maximum_contraction * discrepancy_norm_sq
    required_adjustment = (minimum_dot - correction_dot).clamp_min(0.0)

    temporal_strength = discrepancy_norm_sq.squeeze(-1).squeeze(-1)
    protected_rows = max(1, math.ceil(protected_fraction * temporal_strength.shape[1]))
    protected_indices = torch.topk(
        temporal_strength,
        k=protected_rows,
        dim=1,
        largest=True,
        sorted=False,
    ).indices
    mask = torch.zeros_like(temporal_strength, dtype=torch.bool)
    mask.scatter_(1, protected_indices, True)
    coefficient = torch.where(
        mask[:, :, None, None],
        required_adjustment / discrepancy_norm_sq.clamp_min(eps),
        torch.zeros_like(required_adjustment),
    )
    return (projected + coefficient * discrepancy).to(dtype=correction.dtype)


def trainable_parameter_count(model: nn.Module) -> int:
    """Count trainable parameters for immutable model cards."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
