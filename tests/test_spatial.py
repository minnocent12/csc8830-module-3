from __future__ import annotations

import cv2
import numpy as np
import pytest

from module3.kernels import box_kernel, gaussian_kernel
from module3.spatial import (
    apply_average_blur_gray,
    apply_filter2d_bgr,
    apply_filter2d_gray,
    apply_gaussian_blur_gray,
    validate_kernel,
)


def test_filter2d_preserves_shape_and_impulse_response() -> None:
    image = np.zeros((9, 9), dtype=np.float64)
    image[4, 4] = 1.0
    kernel = box_kernel(3)

    filtered = apply_filter2d_gray(image, kernel, border_type=cv2.BORDER_CONSTANT)
    assert filtered.shape == image.shape
    assert np.allclose(filtered[3:6, 3:6], kernel)


def test_opencv_average_and_gaussian_cross_checks_match_explicit_kernels() -> None:
    image = np.arange(64, dtype=np.float64).reshape(8, 8)

    average_explicit = apply_filter2d_gray(image, box_kernel(5), border_type=cv2.BORDER_CONSTANT)
    average_cv = apply_average_blur_gray(image, 5, border_type=cv2.BORDER_CONSTANT)
    assert np.allclose(average_explicit, average_cv)

    gaussian_explicit = apply_filter2d_gray(image, gaussian_kernel(5), border_type=cv2.BORDER_CONSTANT)
    gaussian_cv = apply_gaussian_blur_gray(image, 5, border_type=cv2.BORDER_CONSTANT)
    assert np.allclose(gaussian_explicit, gaussian_cv)


def test_correlation_and_convolution_differ_for_asymmetric_kernel() -> None:
    image = np.zeros((7, 7), dtype=np.float64)
    image[3, 3] = 1.0
    asymmetric = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    correlation = apply_filter2d_gray(image, asymmetric, border_type=cv2.BORDER_CONSTANT)
    convolution = apply_filter2d_gray(image, np.flipud(np.fliplr(asymmetric)), border_type=cv2.BORDER_CONSTANT)
    assert not np.allclose(correlation, convolution)


def test_per_channel_bgr_filtering_matches_independent_gray_channels() -> None:
    bgr = np.dstack(
        [
            np.arange(25, dtype=np.uint8).reshape(5, 5),
            np.full((5, 5), 100, dtype=np.uint8),
            np.flipud(np.arange(25, dtype=np.uint8).reshape(5, 5)),
        ]
    )
    kernel = box_kernel(3)
    filtered = apply_filter2d_bgr(bgr, kernel, border_type=cv2.BORDER_CONSTANT)
    for channel in range(3):
        expected = apply_filter2d_gray(bgr[:, :, channel], kernel, border_type=cv2.BORDER_CONSTANT)
        assert np.allclose(filtered[:, :, channel], expected)


def test_invalid_kernel_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_kernel(np.ones((2, 2)))
