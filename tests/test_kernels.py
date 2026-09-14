from __future__ import annotations

import numpy as np
import pytest

from module3.kernels import box_kernel, gaussian_kernel, make_kernel, validate_odd_size


def test_box_kernel_is_normalized_and_symmetric() -> None:
    kernel = box_kernel(5)
    assert kernel.shape == (5, 5)
    assert kernel.dtype == np.float64
    assert np.isclose(kernel.sum(), 1.0)
    assert np.allclose(kernel, np.flipud(np.fliplr(kernel)))


def test_gaussian_kernel_is_normalized_and_symmetric() -> None:
    kernel = gaussian_kernel(7)
    assert kernel.shape == (7, 7)
    assert np.isclose(kernel.sum(), 1.0)
    assert np.allclose(kernel, kernel.T)
    assert np.allclose(kernel, np.flipud(np.fliplr(kernel)))


def test_invalid_kernel_sizes_are_rejected() -> None:
    for size in (0, -3, 4):
        with pytest.raises(ValueError):
            validate_odd_size(size)


def test_invalid_sigma_is_rejected() -> None:
    with pytest.raises(ValueError):
        gaussian_kernel(3, sigma=-1.0)
    with pytest.raises(ValueError):
        gaussian_kernel(3, sigma=float("nan"))


def test_make_kernel_names_are_canonical() -> None:
    average = make_kernel("box", 3)
    gaussian = make_kernel("gaussian blur", 5)
    assert average.filter_name == "Average Blur"
    assert gaussian.filter_name == "Gaussian Blur"
    with pytest.raises(ValueError):
        make_kernel("median", 3)
