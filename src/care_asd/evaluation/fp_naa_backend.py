"""Training-free FP-NAA baseline pooling and nearest-neighbour backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np


def _typed_array(value: object) -> np.ndarray:
    """Bridge NumPy stub return differences across supported Python versions."""
    return cast(np.ndarray, value)


@dataclass(frozen=True)
class VarianceRescaler:
    """Normal-only analytical score-rescaling parameters."""

    alpha: float
    reference_bias: np.ndarray
    neighbors: int


def mean_pool(token_grid: np.ndarray) -> np.ndarray:
    """Temporally pool a ``[time, band, dimension]`` token grid."""
    grid = _token_grid(token_grid)
    return _typed_array(grid.mean(axis=0, dtype=np.float64).astype(np.float32))


def rdp_pool(token_grid: np.ndarray, *, gamma: float = 8.0, eps: float = 1.0e-12) -> np.ndarray:
    """Apply relative-deviation pooling while retaining the frequency-patch axis.

    RDP weights each temporal row after concatenating its frequency patches, as in the published
    frequency-preserving representation. The same temporal weight is consequently applied to all
    frequency bands.
    """
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    grid = _token_grid(token_grid).astype(np.float64, copy=False)
    flattened = grid.reshape(grid.shape[0], -1)
    temporal_mean = flattened.mean(axis=0)
    deviations = np.linalg.norm(flattened - temporal_mean, axis=1)
    maximum = float(deviations.max(initial=0.0))
    if maximum <= eps:
        weights = np.full(grid.shape[0], 1.0 / grid.shape[0], dtype=np.float64)
    else:
        relative = deviations / maximum
        unnormalized = np.power(1.0 + relative, gamma)
        weights = unnormalized / unnormalized.sum()
    return cast(
        np.ndarray,
        np.einsum("t,tbd->bd", weights, grid, optimize=True).astype(np.float32),
    )


def global_average_descriptor(token_grid: np.ndarray) -> np.ndarray:
    """Return the conventional global average BEATs descriptor."""
    grid = _token_grid(token_grid)
    return _typed_array(grid.mean(axis=(0, 1), dtype=np.float64).astype(np.float32))


def cosine_distance_matrix(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Return ``0.5 * (1 - cosine)`` for all query/reference pairs."""
    query = _matrix(queries, "queries").astype(np.float64, copy=False)
    reference = _matrix(references, "references").astype(np.float64, copy=False)
    if query.shape[1] != reference.shape[1]:
        raise ValueError("queries and references must have the same feature dimension")
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    query_unit = np.divide(query, np.maximum(query_norm, eps))
    reference_unit = np.divide(reference, np.maximum(reference_norm, eps))
    similarity = np.clip(query_unit @ reference_unit.T, -1.0, 1.0)
    return cast(np.ndarray, (0.5 * (1.0 - similarity)).astype(np.float64))


def fit_variance_rescaler(
    references: np.ndarray,
    *,
    neighbors: int = 4,
    eps: float = 1.0e-12,
) -> VarianceRescaler:
    """Fit the analytical normal-score variance-minimization coefficient.

    Each reference is scored leave-one-out. The local bias is the mean distance to its K nearest
    other references. Alpha is ``Cov(raw_score, matched_bias) / Var(matched_bias)``.
    """
    reference = _matrix(references, "references")
    if len(reference) < 2:
        raise ValueError("At least two references are required")
    if not 1 <= neighbors < len(reference):
        raise ValueError("neighbors must be in [1, number_of_references - 1]")
    distances = cosine_distance_matrix(reference, reference, eps=eps)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    raw = distances[np.arange(len(reference)), nearest]
    local = np.partition(distances, neighbors - 1, axis=1)[:, :neighbors].mean(axis=1)
    matched_bias = local[nearest]
    centered_bias = matched_bias - matched_bias.mean()
    variance = float(np.mean(centered_bias**2))
    if variance <= eps:
        alpha = 0.0
    else:
        covariance = float(np.mean((raw - raw.mean()) * centered_bias))
        alpha = covariance / variance
    return VarianceRescaler(alpha=alpha, reference_bias=local, neighbors=neighbors)


def variance_rescaled_knn_scores(
    queries: np.ndarray,
    references: np.ndarray,
    rescaler: VarianceRescaler,
    *,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Score queries with the published subtractive local-bias correction."""
    reference = _matrix(references, "references")
    if rescaler.reference_bias.shape != (len(reference),):
        raise ValueError("reference_bias does not match references")
    distances = cosine_distance_matrix(queries, reference, eps=eps)
    corrected = distances - rescaler.alpha * rescaler.reference_bias[None, :]
    return cast(np.ndarray, corrected.min(axis=1))


def beam_scores(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    neighbors: int = 4,
    variance_rescaling: bool = True,
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, tuple[VarianceRescaler, ...]]:
    """Band-aligned BEAM retrieval with uniform score aggregation."""
    query = _band_descriptors(queries, "queries")
    reference = _band_descriptors(references, "references")
    if query.shape[1:] != reference.shape[1:]:
        raise ValueError("queries and references must share band and feature dimensions")
    band_scores: list[np.ndarray] = []
    rescalers: list[VarianceRescaler] = []
    for band in range(query.shape[1]):
        query_band = query[:, band, :]
        reference_band = reference[:, band, :]
        if variance_rescaling:
            rescaler = fit_variance_rescaler(
                reference_band,
                neighbors=neighbors,
                eps=eps,
            )
            score = variance_rescaled_knn_scores(
                query_band,
                reference_band,
                rescaler,
                eps=eps,
            )
        else:
            bias = np.zeros(len(reference_band), dtype=np.float64)
            rescaler = VarianceRescaler(alpha=0.0, reference_bias=bias, neighbors=neighbors)
            score = cosine_distance_matrix(query_band, reference_band, eps=eps).min(axis=1)
        rescalers.append(rescaler)
        band_scores.append(score)
    return np.stack(band_scores, axis=1).mean(axis=1), tuple(rescalers)


def global_knn_scores(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    neighbors: int = 4,
    variance_rescaling: bool = True,
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, VarianceRescaler]:
    """Score flat descriptors with the same normal-only calibration as BEAM."""
    query = _matrix(queries, "queries")
    reference = _matrix(references, "references")
    if variance_rescaling:
        rescaler = fit_variance_rescaler(reference, neighbors=neighbors, eps=eps)
        scores = variance_rescaled_knn_scores(query, reference, rescaler, eps=eps)
    else:
        rescaler = VarianceRescaler(
            alpha=0.0,
            reference_bias=np.zeros(len(reference), dtype=np.float64),
            neighbors=neighbors,
        )
        scores = cosine_distance_matrix(query, reference, eps=eps).min(axis=1)
    return scores, rescaler


def accelerated_beam_scores(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    neighbors: int = 4,
    device: str = "cuda",
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU/torch implementation of variance-rescaled BEAM.

    Returns query scores and one analytically fitted alpha per frequency band. Torch is imported
    lazily so the core package and unit tests do not require the optional neural dependency.
    """
    import torch
    import torch.nn.functional as functional

    query_array = _band_descriptors(queries, "queries")
    reference_array = _band_descriptors(references, "references")
    if query_array.shape[1:] != reference_array.shape[1:]:
        raise ValueError("queries and references must share band and feature dimensions")
    if not 1 <= neighbors < len(reference_array):
        raise ValueError("neighbors must be in [1, number_of_references - 1]")
    query = torch.as_tensor(query_array, dtype=torch.float32, device=device).permute(1, 0, 2)
    reference = torch.as_tensor(reference_array, dtype=torch.float32, device=device).permute(1, 0, 2)
    query = functional.normalize(query, p=2.0, dim=-1, eps=eps)
    reference = functional.normalize(reference, p=2.0, dim=-1, eps=eps)
    with torch.inference_mode():
        reference_distance = 0.5 * (
            1.0 - torch.bmm(reference, reference.transpose(1, 2)).clamp(-1.0, 1.0)
        )
        diagonal = torch.arange(reference.shape[1], device=device)
        reference_distance[:, diagonal, diagonal] = torch.inf
        local_bias = torch.topk(
            reference_distance,
            k=neighbors,
            dim=2,
            largest=False,
        ).values.mean(dim=2)
        raw, matched_index = reference_distance.min(dim=2)
        matched_bias = torch.gather(local_bias, 1, matched_index)
        centered_raw = raw - raw.mean(dim=1, keepdim=True)
        centered_bias = matched_bias - matched_bias.mean(dim=1, keepdim=True)
        variance = centered_bias.square().mean(dim=1)
        covariance = (centered_raw * centered_bias).mean(dim=1)
        alpha = torch.where(variance > eps, covariance / variance, torch.zeros_like(variance))
        query_distance = 0.5 * (
            1.0 - torch.bmm(query, reference.transpose(1, 2)).clamp(-1.0, 1.0)
        )
        corrected = query_distance - alpha[:, None, None] * local_bias[:, None, :]
        scores = corrected.min(dim=2).values.mean(dim=0)
    return scores.cpu().numpy().astype(np.float64), alpha.cpu().numpy().astype(np.float64)


def accelerated_global_knn_scores(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    neighbors: int = 4,
    device: str = "cuda",
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, float]:
    """GPU/torch implementation of flat variance-rescaled nearest-neighbour scoring."""
    query = _matrix(queries, "queries")[:, None, :]
    reference = _matrix(references, "references")[:, None, :]
    scores, alpha = accelerated_beam_scores(
        query,
        reference,
        neighbors=neighbors,
        device=device,
        eps=eps,
    )
    return scores, float(alpha[0])


def accelerated_rdp_pool(
    token_grids: np.ndarray,
    *,
    gamma: float,
    device: str = "cuda",
    batch_size: int = 64,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Batch RDP on torch while retaining ``[item, band, dimension]`` output."""
    import torch

    values = np.asarray(token_grids)
    if values.ndim != 4 or min(values.shape) < 1:
        raise ValueError("token_grids must have shape [items, time, band, dimension]")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("token_grids must contain finite numeric values")
    if gamma < 0.0 or batch_size < 1:
        raise ValueError("gamma must be non-negative and batch_size must be positive")
    pooled: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            grid = torch.as_tensor(
                values[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            flattened = grid.flatten(start_dim=2)
            temporal_mean = flattened.mean(dim=1, keepdim=True)
            deviations = torch.linalg.vector_norm(flattened - temporal_mean, dim=2)
            maximum = deviations.amax(dim=1, keepdim=True)
            relative = torch.where(maximum > eps, deviations / maximum.clamp_min(eps), 0.0)
            weights = (1.0 + relative).pow(gamma)
            weights = weights / weights.sum(dim=1, keepdim=True)
            result = torch.einsum("nt,ntbd->nbd", weights, grid)
            pooled.append(result.cpu().numpy().astype(np.float32))
    return np.concatenate(pooled, axis=0)


def _token_grid(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or min(array.shape) < 1:
        raise ValueError("token_grid must have shape [time, band, dimension]")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("token_grid must contain finite numeric values")
    return array


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must have shape [items, dimension]")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def _band_descriptors(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or min(array.shape) < 1:
        raise ValueError(f"{name} must have shape [items, bands, dimension]")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return array
