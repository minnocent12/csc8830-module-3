from __future__ import annotations

import cv2
import numpy as np

from module3.fourier import fourier_filter_gray, wrong_unaligned_fourier_filter_gray
from module3.kernels import box_kernel, gaussian_kernel
from module3.spatial import apply_filter2d_gray


def _synthetic_gray() -> np.ndarray:
    y, x = np.mgrid[0:24, 0:32]
    return (3 * x + 5 * y + 40 * ((x // 5 + y // 4) % 2)).astype(np.float64)


def test_fourier_filter_matches_zero_boundary_spatial_for_average_and_gaussian() -> None:
    image = _synthetic_gray()
    for kernel in (box_kernel(3), box_kernel(7), gaussian_kernel(5)):
        spatial = apply_filter2d_gray(image, kernel, border_type=cv2.BORDER_CONSTANT)
        fourier = fourier_filter_gray(image, kernel)
        assert fourier.image.shape == image.shape
        assert fourier.max_imaginary_abs < 1e-10
        assert np.max(np.abs(spatial - fourier.image)) < 1e-10


def test_impulse_response_matches_kernel_at_center() -> None:
    image = np.zeros((11, 11), dtype=np.float64)
    image[5, 5] = 1.0
    kernel = gaussian_kernel(5)
    result = fourier_filter_gray(image, kernel).image
    assert np.allclose(result[3:8, 3:8], kernel, atol=1e-12)


def test_unaligned_kernel_embedding_is_measurably_wrong() -> None:
    image = _synthetic_gray()
    kernel = box_kernel(5)
    correct = fourier_filter_gray(image, kernel).image
    wrong = wrong_unaligned_fourier_filter_gray(image, kernel)
    assert np.max(np.abs(correct - wrong)) > 1.0
