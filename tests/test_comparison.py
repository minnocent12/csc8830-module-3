from __future__ import annotations

import numpy as np

from module3.comparison import compare_spatial_fourier
from module3.kernels import box_kernel, gaussian_kernel


def test_comparison_pipeline_reports_machine_precision_error() -> None:
    y, x = np.mgrid[0:20, 0:28]
    image = (x * 7 + y * 11 + ((x + y) % 3) * 30).astype(np.float64)
    for kernel in (box_kernel(3), box_kernel(9), gaussian_kernel(5), gaussian_kernel(9)):
        result = compare_spatial_fourier(image, kernel)
        assert result.original.shape == image.shape
        assert result.difference.shape == image.shape
        assert result.metrics.mae < 1e-9
        assert result.metrics.mse < 1e-15
        assert result.max_imaginary_abs < 1e-10
