from __future__ import annotations

import numpy as np
import pytest

from care_asd.evaluation.fp_naa_backend import (
    beam_scores,
    cosine_distance_matrix,
    fit_variance_rescaler,
    mean_pool,
    rdp_pool,
    variance_rescaled_knn_scores,
)


def test_rdp_gamma_zero_is_temporal_mean() -> None:
    rng = np.random.default_rng(42)
    tokens = rng.normal(size=(5, 3, 4)).astype(np.float32)
    np.testing.assert_allclose(rdp_pool(tokens, gamma=0.0), mean_pool(tokens), atol=1.0e-6)


def test_rdp_emphasizes_deviating_temporal_row() -> None:
    tokens = np.zeros((3, 1, 2), dtype=np.float32)
    tokens[2, 0] = [9.0, 0.0]
    pooled_mean = rdp_pool(tokens, gamma=0.0)
    pooled_rdp = rdp_pool(tokens, gamma=8.0)
    assert pooled_rdp[0, 0] > pooled_mean[0, 0]


def test_cosine_distance_has_expected_scale() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    reference = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
    result = cosine_distance_matrix(query, reference)
    np.testing.assert_allclose(result, [[0.0, 1.0], [0.5, 0.5]], atol=1.0e-12)


def test_beam_can_select_a_different_reference_per_band() -> None:
    references = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    query = np.asarray([[[1.0, 0.0], [1.0, 0.0]]])
    scores, _ = beam_scores(
        query,
        references,
        neighbors=1,
        variance_rescaling=False,
    )
    assert scores[0] == pytest.approx(0.0)


def test_variance_rescaler_reduces_training_normal_score_variance() -> None:
    references = np.asarray(
        [[1.0, 0.0], [0.99, 0.1], [0.7, 0.7], [0.68, 0.73], [0.0, 1.0]],
        dtype=np.float64,
    )
    rescaler = fit_variance_rescaler(references, neighbors=1)
    raw_distance = cosine_distance_matrix(references, references)
    np.fill_diagonal(raw_distance, np.inf)
    raw = raw_distance.min(axis=1)
    corrected = []
    for index in range(len(references)):
        keep = np.arange(len(references)) != index
        subset = references[keep]
        subset_bias = rescaler.reference_bias[keep]
        query_distance = cosine_distance_matrix(references[index : index + 1], subset)
        corrected.append(float((query_distance - rescaler.alpha * subset_bias[None, :]).min()))
    assert np.var(corrected) <= np.var(raw) + 1.0e-12


def test_variance_rescaled_scores_validate_bias_shape() -> None:
    references = np.eye(3)
    rescaler = fit_variance_rescaler(references, neighbors=1)
    broken = type(rescaler)(alpha=rescaler.alpha, reference_bias=np.zeros(2), neighbors=1)
    with pytest.raises(ValueError, match="reference_bias"):
        variance_rescaled_knn_scores(references, references, broken)
