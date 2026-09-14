from __future__ import annotations

import numpy as np
import pytest

from module3.metrics import compute_error_metrics


def test_metrics_match_hand_computed_values() -> None:
    spatial = np.array([[1.0, 2.0], [3.0, 4.0]])
    fourier = np.array([[1.0, 4.0], [1.0, 4.0]])
    metrics = compute_error_metrics(spatial, fourier)
    assert metrics.mae == pytest.approx(1.0)
    assert metrics.mse == pytest.approx(2.0)
    assert metrics.rmse == pytest.approx(np.sqrt(2.0))
    assert metrics.max_abs_error == pytest.approx(2.0)


def test_psnr_is_infinite_for_identical_arrays() -> None:
    image = np.ones((3, 3), dtype=np.float64)
    assert compute_error_metrics(image, image).psnr_db == float("inf")


def test_metric_input_validation() -> None:
    with pytest.raises(ValueError):
        compute_error_metrics(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        compute_error_metrics(np.array([[np.nan]]), np.array([[0.0]]))
