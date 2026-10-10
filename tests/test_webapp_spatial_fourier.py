"""Contract tests for the Module 3 Spatial vs Fourier page after Phase 13 migration.

Covers:
1. Default/bundled state: header, subheaders, widgets, images, metrics
2. Widget contract: keys, labels, defaults, options
3. Metric contract: 5 metrics, exact labels, baseline values
4. Caption contract: methodology and imaginary-residual captions
5. Alternate filter and kernel size states
6. No dataframes introduced
7. Public copy: no em/en dashes, no internal paths, no phase terminology
8. No exception in any tested state
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

APP = str(ROOT / "app.py")

EXPECTED_SUBHEADERS = ["Input", "Configuration", "Comparison Results"]

# Baseline metric values at default state (Average Blur 5x5, bundled sample)
BASELINE_METRICS = {
    "MAE": "4.663e-14",
    "MSE": "3.690e-27",
    "RMSE": "6.075e-14",
    "Max error": "2.558e-13",
    "PSNR (dB)": "312.46",
}


@pytest.fixture(scope="module")
def at_default() -> AppTest:
    """AppTest with Spatial vs Fourier page at default state."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Spatial vs Fourier").run()
    return at


# ── 1. No exception ───────────────────────────────────────────────────────────

def test_default_no_exception(at_default: AppTest) -> None:
    assert not at_default.exception


# ── 2. Page header ────────────────────────────────────────────────────────────

def test_default_header_value(at_default: AppTest) -> None:
    assert any(h.value == "Spatial vs Fourier" for h in at_default.header)


def test_default_title_once(at_default: AppTest) -> None:
    count = sum(1 for h in at_default.header if h.value == "Spatial vs Fourier")
    assert count == 1


# ── 3. Section structure ──────────────────────────────────────────────────────

def test_default_subheaders_present(at_default: AppTest) -> None:
    sub_vals = [s.value for s in at_default.subheader]
    for expected in EXPECTED_SUBHEADERS:
        assert expected in sub_vals, f"Missing subheader: {expected!r} (got {sub_vals})"


def test_default_subheaders_order(at_default: AppTest) -> None:
    sub_vals = [s.value for s in at_default.subheader]
    indices = [sub_vals.index(s) for s in EXPECTED_SUBHEADERS if s in sub_vals]
    assert indices == sorted(indices), f"Subheaders out of order: {sub_vals}"


def test_default_exactly_three_subheaders(at_default: AppTest) -> None:
    sub_vals = [s.value for s in at_default.subheader]
    assert sub_vals == EXPECTED_SUBHEADERS, f"Got {sub_vals}"


# ── 4. Input widget contract ──────────────────────────────────────────────────

def test_default_file_uploader_present(at_default: AppTest) -> None:
    if not hasattr(at_default, "file_uploader"):
        pytest.skip("file_uploader not tracked by this AppTest version")
    assert len(at_default.file_uploader) == 1


def test_default_file_uploader_label(at_default: AppTest) -> None:
    if not hasattr(at_default, "file_uploader") or not at_default.file_uploader:
        pytest.skip("file_uploader not tracked by this AppTest version")
    fu = at_default.file_uploader[0]
    label = getattr(fu, "label", None) or getattr(fu.proto, "label", None)
    if label is None:
        pytest.skip("file_uploader label not accessible in this AppTest version")
    assert label == "Image for comparison"


# ── 5. Configuration widget contract ─────────────────────────────────────────

def test_default_selectbox_present(at_default: AppTest) -> None:
    assert len(at_default.selectbox) == 1


def test_default_selectbox_label(at_default: AppTest) -> None:
    assert at_default.selectbox[0].label == "Filter"


def test_default_selectbox_value(at_default: AppTest) -> None:
    assert at_default.selectbox[0].value == "Average Blur"


def test_default_selectbox_options(at_default: AppTest) -> None:
    assert at_default.selectbox[0].options == ["Average Blur", "Gaussian Blur"]


def test_default_select_slider_present(at_default: AppTest) -> None:
    assert len(at_default.select_slider) == 1


def test_default_select_slider_label(at_default: AppTest) -> None:
    assert at_default.select_slider[0].label == "Kernel size"


def test_default_select_slider_value(at_default: AppTest) -> None:
    assert at_default.select_slider[0].value == 5


def test_default_select_slider_options(at_default: AppTest) -> None:
    assert list(at_default.select_slider[0].options) == ["3", "5", "7", "9", "11"]


# ── 6. Comparison Results contract ───────────────────────────────────────────

def test_default_four_images(at_default: AppTest) -> None:
    if not hasattr(at_default, "image"):
        pytest.skip("image not tracked by this AppTest version")
    assert len(at_default.image) == 4


def test_default_no_dataframe(at_default: AppTest) -> None:
    assert len(at_default.dataframe) == 0


def test_default_five_metrics(at_default: AppTest) -> None:
    assert len(at_default.metric) == 5


def test_default_metric_labels(at_default: AppTest) -> None:
    labels = [m.label for m in at_default.metric]
    assert "MAE" in labels
    assert "MSE" in labels
    assert "RMSE" in labels
    assert "Max error" in labels
    assert "PSNR (dB)" in labels


def test_default_metric_labels_order(at_default: AppTest) -> None:
    labels = [m.label for m in at_default.metric]
    assert labels == ["MAE", "MSE", "RMSE", "Max error", "PSNR (dB)"], f"Got {labels}"


def test_default_metric_mae_value(at_default: AppTest) -> None:
    mae = next(m.value for m in at_default.metric if m.label == "MAE")
    assert mae == BASELINE_METRICS["MAE"], f"MAE={mae!r}"


def test_default_metric_psnr_value(at_default: AppTest) -> None:
    psnr = next(m.value for m in at_default.metric if m.label == "PSNR (dB)")
    assert psnr == BASELINE_METRICS["PSNR (dB)"], f"PSNR={psnr!r}"


def test_default_metric_max_error_value(at_default: AppTest) -> None:
    mx = next(m.value for m in at_default.metric if m.label == "Max error")
    assert mx == BASELINE_METRICS["Max error"], f"Max error={mx!r}"


# ── 7. Caption contract ───────────────────────────────────────────────────────

def test_default_methodology_caption(at_default: AppTest) -> None:
    captions = [c.value for c in at_default.caption]
    assert any("convolution theorem" in c for c in captions), f"Methodology caption not found: {captions}"


def test_default_imaginary_residual_caption(at_default: AppTest) -> None:
    captions = [c.value for c in at_default.caption]
    assert any("imaginary" in c for c in captions), f"Imaginary caption not found: {captions}"


# ── 8. No download actions ────────────────────────────────────────────────────

def test_default_no_download_button(at_default: AppTest) -> None:
    if not hasattr(at_default, "download_button"):
        pytest.skip("download_button not tracked by this AppTest version")
    assert len(at_default.download_button) == 0


# ── 9. Alternate filter state ─────────────────────────────────────────────────

def test_gaussian_filter_no_exception() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Spatial vs Fourier").run()
    at.selectbox[0].set_value("Gaussian Blur").run()
    assert not at.exception


def test_gaussian_filter_five_metrics() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Spatial vs Fourier").run()
    at.selectbox[0].set_value("Gaussian Blur").run()
    assert len(at.metric) == 5


def test_gaussian_filter_four_images() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Spatial vs Fourier").run()
    at.selectbox[0].set_value("Gaussian Blur").run()
    if not hasattr(at, "image"):
        pytest.skip("image not tracked by this AppTest version")
    assert len(at.image) == 4


# ── 10. Alternate kernel size ─────────────────────────────────────────────────

def test_kernel_size_9_no_exception() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Spatial vs Fourier").run()
    at.select_slider[0].set_value(9).run()
    assert not at.exception


def test_kernel_size_9_five_metrics() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Spatial vs Fourier").run()
    at.select_slider[0].set_value(9).run()
    assert len(at.metric) == 5


def test_kernel_size_9_four_images() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Spatial vs Fourier").run()
    at.select_slider[0].set_value(9).run()
    if not hasattr(at, "image"):
        pytest.skip("image not tracked by this AppTest version")
    assert len(at.image) == 4


def test_kernel_size_3_psnr_high() -> None:
    """Smaller kernel produces higher PSNR (tighter equivalence) for Average Blur."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Spatial vs Fourier").run()
    at.select_slider[0].set_value(3).run()
    psnr = next((m.value for m in at.metric if m.label == "PSNR (dB)"), None)
    assert psnr is not None and psnr != "inf"
    # PSNR should be high (> 200 dB) for both small and large kernels given near-exact match
    assert float(psnr) > 200.0


# ── 11. Numerical contract at baseline ───────────────────────────────────────

def test_baseline_all_metric_values(at_default: AppTest) -> None:
    """All 5 baseline metric values match the pre-migration contract exactly."""
    actual = {m.label: m.value for m in at_default.metric}
    for label, expected in BASELINE_METRICS.items():
        assert actual.get(label) == expected, f"{label}: expected {expected!r}, got {actual.get(label)!r}"


# ── 12. Public copy ───────────────────────────────────────────────────────────

def _all_visible_text(at: AppTest) -> str:
    parts = []
    for attr in ("header", "subheader", "markdown", "caption", "info", "warning", "error"):
        for el in getattr(at, attr):
            parts.append(str(el.value))
    return " ".join(parts)


def test_no_em_dash(at_default: AppTest) -> None:
    assert "—" not in _all_visible_text(at_default)


def test_no_en_dash(at_default: AppTest) -> None:
    assert "–" not in _all_visible_text(at_default)


def test_no_internal_path(at_default: AppTest) -> None:
    text = _all_visible_text(at_default)
    assert "module3_sample.png" not in text
    assert "pages.py" not in text


def test_no_phase_terminology(at_default: AppTest) -> None:
    text = _all_visible_text(at_default)
    for term in ["Phase 13", "Phase 12", "migration", "AGENTS.md"]:
        assert term not in text, f"Internal term {term!r} in visible text"


# ── 13. Upload state contract ─────────────────────────────────────────────────

def _make_synthetic_png() -> bytes:
    """Return a deterministic 64x96 BGR PNG as bytes (gradient+checkerboard)."""
    import cv2, numpy as np
    h, w = 64, 96
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.linspace(30, 200, w, dtype=np.uint8), (h, 1))
    img[:, :, 1] = np.tile(np.linspace(80, 180, h, dtype=np.uint8)[:, None], (1, w))
    img[:, :, 2] = 120
    for r in range(h):
        for c in range(w):
            if ((r // 8) + (c // 8)) % 2:
                img[r, c] = np.clip(img[r, c].astype(int) + 40, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode('.png', img)
    return enc.tobytes()


_SYNTHETIC_PNG = _make_synthetic_png()
_UPLOAD_TUPLE = ("test_synthetic.png", _SYNTHETIC_PNG, "image/png")

# Pre/post upload metric contract at default settings (Average Blur 5x5)
UPLOAD_BASELINE_METRICS = {
    "MAE": "2.949e-14",
    "MSE": "1.501e-27",
    "RMSE": "3.875e-14",
    "Max error": "1.705e-13",
    "PSNR (dB)": "316.37",
}


@pytest.fixture(scope="module")
def at_upload() -> AppTest:
    """AppTest with Spatial vs Fourier page and synthetic image uploaded."""
    if not hasattr(AppTest.from_file(APP, default_timeout=60).run(), "file_uploader"):
        pytest.skip("file_uploader not tracked by this AppTest version")
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Spatial vs Fourier").run()
    at.file_uploader[0].set_value(_UPLOAD_TUPLE).run()
    return at


def test_upload_no_exception(at_upload: AppTest) -> None:
    assert not at_upload.exception


def test_upload_four_images(at_upload: AppTest) -> None:
    if not hasattr(at_upload, "image"):
        pytest.skip("image not tracked by this AppTest version")
    assert len(at_upload.image) == 4


def test_upload_subheaders_unchanged(at_upload: AppTest) -> None:
    sub_vals = [s.value for s in at_upload.subheader]
    assert sub_vals == EXPECTED_SUBHEADERS, f"Got {sub_vals}"


def test_upload_five_metrics(at_upload: AppTest) -> None:
    assert len(at_upload.metric) == 5


def test_upload_metric_labels_order(at_upload: AppTest) -> None:
    labels = [m.label for m in at_upload.metric]
    assert labels == ["MAE", "MSE", "RMSE", "Max error", "PSNR (dB)"], f"Got {labels}"


def test_upload_metric_values_match_baseline(at_upload: AppTest) -> None:
    """Uploaded-image metric values match pipeline-computed baseline (pre/post identical)."""
    actual = {m.label: m.value for m in at_upload.metric}
    for label, expected in UPLOAD_BASELINE_METRICS.items():
        assert actual.get(label) == expected, (
            f"{label}: expected {expected!r}, got {actual.get(label)!r}"
        )


def test_upload_methodology_caption(at_upload: AppTest) -> None:
    captions = [c.value for c in at_upload.caption]
    assert any("convolution theorem" in c for c in captions)


def test_upload_imaginary_residual_caption(at_upload: AppTest) -> None:
    captions = [c.value for c in at_upload.caption]
    assert any("imaginary" in c for c in captions)
