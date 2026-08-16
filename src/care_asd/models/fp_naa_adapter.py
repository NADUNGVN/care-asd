"""Capacity-matched reference-conditioned adapter for FP-NAA C1/C2/C3."""

from __future__ import annotations

import torch
from torch import Tensor, nn


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
    ) -> None:
        super().__init__()
        if min(embedding_dim, hidden_dim, attention_heads) < 1:
            raise ValueError("embedding_dim, hidden_dim, and attention_heads must be positive")
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.embedding_dim = embedding_dim
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
        query = self.target_projection(self.target_norm(target_rows))
        key_value = self.reference_projection(self.reference_norm(reference_rows))
        attended, _ = self.cross_attention(
            query,
            key_value,
            key_value,
            need_weights=False,
        )
        correction = self.fusion(torch.cat((query, attended), dim=-1))
        adapted = target_rows + correction
        return adapted.reshape(batch, bands, time, dimension).permute(0, 2, 1, 3).contiguous()


def trainable_parameter_count(model: nn.Module) -> int:
    """Count trainable parameters for immutable model cards."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

