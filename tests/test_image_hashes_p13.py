"""Image hash contract for Phase 13 migration.

Directly calls the processing pipeline (not through Streamlit) to verify that
the exact same numpy arrays are produced before and after the presentation migration.
These hashes were captured before any Phase 13 editing from the live codebase.

Short hashes (first 16 hex chars of SHA-256) are used for readability; they are
sufficient for collision-free identification of 256x384 image arrays.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from module3.io_utils import bgr_to_gray_float64, load_image_bgr, to_display_uint8
from module3.kernels import make_kernel
from module3.spatial import apply_filter2d_bgr
from module3.comparison import compare_spatial_fourier
from module3.visualization import contrast_stretch

SAMPLE = ROOT / "data" / "sample_images" / "module3_sample.png"
pytestmark = pytest.mark.skipif(
    not SAMPLE.is_file(), reason="bundled sample not available"
)


def sha16(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def blurring_images(filter_key: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (orig_rgb, filtered_rgb) as show_bgr_image would produce them."""
    image = load_image_bgr(SAMPLE)
    ks = make_kernel(filter_key, size)
    filtered = apply_filter2d_bgr(image, ks.kernel, border_type=cv2.BORDER_REFLECT_101)
    orig_rgb = cv2.cvtColor(to_display_uint8(image), cv2.COLOR_BGR2RGB)
    filt_rgb = cv2.cvtColor(to_display_uint8(filtered), cv2.COLOR_BGR2RGB)
    return orig_rgb, filt_rgb


# Hashes pinned from pre-migration baseline run (same code path, same sample).
BLURRING_HASHES: dict[tuple[str, int], tuple[str, str]] = {
    ("average", 3): ("2f63e07813c44e2a", "d2469062708b800d"),
    ("average", 5): ("2f63e07813c44e2a", "c07c86654f590045"),
    ("average", 9): ("2f63e07813c44e2a", "57090a17cf6cc496"),
    ("gaussian", 3): ("2f63e07813c44e2a", "8162ba9df3c610a1"),
    ("gaussian", 5): ("2f63e07813c44e2a", "4104c49fc7869e41"),
    ("gaussian", 9): ("2f63e07813c44e2a", "d5007acf1ad668ae"),
}


@pytest.mark.parametrize("filter_key,size", list(BLURRING_HASHES.keys()))
def test_blurring_image_hashes(filter_key: str, size: int) -> None:
    orig, filt = blurring_images(filter_key, size)
    expected_orig, expected_filt = BLURRING_HASHES[(filter_key, size)]
    assert sha16(orig) == expected_orig, f"orig hash mismatch for {filter_key} {size}x{size}"
    assert sha16(filt) == expected_filt, f"filt hash mismatch for {filter_key} {size}x{size}"


def test_blurring_image_shape() -> None:
    orig, filt = blurring_images("average", 5)
    assert orig.shape == (256, 384, 3)
    assert filt.shape == (256, 384, 3)
    assert orig.dtype == np.uint8
    assert filt.dtype == np.uint8


def test_blurring_images_differ() -> None:
    orig, filt = blurring_images("average", 5)
    assert not np.array_equal(orig, filt), "Filtered image must differ from original"


# Spatial vs Fourier hashes
SVF_HASHES: dict[tuple[str, int], dict[str, str]] = {
    ("average", 3): {
        "orig": "af6269d48ceb14f0",
        "spatial": "6f466e9857374a7e",
        "fourier": "6f466e9857374a7e",
    },
    ("average", 5): {
        "orig": "af6269d48ceb14f0",
        "spatial": "d29ffba13bbd3b14",
        "fourier": "d29ffba13bbd3b14",
    },
    ("average", 9): {
        "orig": "af6269d48ceb14f0",
        "spatial": "e8fa48e34de6f27f",
        "fourier": "e8fa48e34de6f27f",
    },
    ("gaussian", 5): {
        "orig": "af6269d48ceb14f0",
        "spatial": "c12dfa0e6edb1f50",
        "fourier": "f1bf30f4423380cd",
    },
}

SVF_METRICS: dict[tuple[str, int], dict[str, float]] = {
    ("average", 5): {"mae": 4.663273e-14, "psnr_db": 312.4603},
    ("average", 3): {"mae": 7.024768e-14, "psnr_db": 309.3020},
    ("gaussian", 5): {"mae": 2.966762e-14, "psnr_db": 316.0253},
}


@pytest.mark.parametrize("filter_key,size", list(SVF_HASHES.keys()))
def test_svf_image_hashes(filter_key: str, size: int) -> None:
    image = load_image_bgr(SAMPLE)
    gray = bgr_to_gray_float64(image)
    ks = make_kernel(filter_key, size)
    result = compare_spatial_fourier(gray, ks.kernel)
    hashes = SVF_HASHES[(filter_key, size)]
    assert sha16(to_display_uint8(result.original)) == hashes["orig"]
    assert sha16(to_display_uint8(result.spatial)) == hashes["spatial"]
    assert sha16(to_display_uint8(result.fourier)) == hashes["fourier"]


@pytest.mark.parametrize("filter_key,size", list(SVF_METRICS.keys()))
def test_svf_metric_values(filter_key: str, size: int) -> None:
    image = load_image_bgr(SAMPLE)
    gray = bgr_to_gray_float64(image)
    ks = make_kernel(filter_key, size)
    result = compare_spatial_fourier(gray, ks.kernel)
    expected = SVF_METRICS[(filter_key, size)]
    assert result.metrics.mae == pytest.approx(expected["mae"], rel=1e-5)
    assert result.metrics.psnr_db == pytest.approx(expected["psnr_db"], rel=1e-4)
