"""Contract tests for the Module 3 Image Blurring page after Phase 13 migration.

Covers:
1. Default/bundled state: header, subheaders, widgets, images, captions
2. Widget contract: keys, labels, defaults, options
3. Alternate filter state (Gaussian Blur)
4. Alternate kernel size state (9x9)
5. Image and dataframe counts
6. No metrics introduced
7. No download actions
8. Public copy: no em/en dashes, no internal paths, no phase terminology
9. No exception in any tested state
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

APP = str(ROOT / "app.py")

EXPECTED_SUBHEADERS = ["Input", "Configuration", "Results"]


@pytest.fixture(scope="module")
def at_default() -> AppTest:
    """AppTest with Image Blurring page at default state."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Image Blurring").run()
    return at


# ── 1. No exception ───────────────────────────────────────────────────────────

def test_default_no_exception(at_default: AppTest) -> None:
    assert not at_default.exception


# ── 2. Page header ────────────────────────────────────────────────────────────

def test_default_header_value(at_default: AppTest) -> None:
    assert any(h.value == "Image Blurring" for h in at_default.header)


def test_default_title_once(at_default: AppTest) -> None:
    count = sum(1 for h in at_default.header if h.value == "Image Blurring")
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
    assert label == "Image"


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


# ── 6. Results contract ───────────────────────────────────────────────────────

def test_default_two_images(at_default: AppTest) -> None:
    if not hasattr(at_default, "image"):
        pytest.skip("image not tracked by this AppTest version")
    assert len(at_default.image) == 2


def test_default_one_dataframe(at_default: AppTest) -> None:
    """Kernel dataframe is present."""
    assert len(at_default.dataframe) == 1


def test_default_no_metrics(at_default: AppTest) -> None:
    assert len(at_default.metric) == 0


def test_default_kernel_caption_present(at_default: AppTest) -> None:
    captions = [c.value for c in at_default.caption]
    assert any("sum to" in c for c in captions), f"Kernel caption not found in {captions}"


def test_default_border_caption_present(at_default: AppTest) -> None:
    captions = [c.value for c in at_default.caption]
    assert any("BORDER_REFLECT_101" in c for c in captions), f"Border caption not found in {captions}"


# ── 7. No download actions ────────────────────────────────────────────────────

def test_default_no_download_button(at_default: AppTest) -> None:
    if not hasattr(at_default, "download_button"):
        pytest.skip("download_button not tracked by this AppTest version")
    assert len(at_default.download_button) == 0


# ── 8. Alternate filter state ─────────────────────────────────────────────────

def test_gaussian_filter_no_exception(at_default: AppTest) -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Image Blurring").run()
    at.selectbox[0].set_value("Gaussian Blur").run()
    assert not at.exception


def test_gaussian_filter_two_images() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Image Blurring").run()
    at.selectbox[0].set_value("Gaussian Blur").run()
    if not hasattr(at, "image"):
        pytest.skip("image not tracked by this AppTest version")
    assert len(at.image) == 2


def test_gaussian_filter_kernel_caption_updated() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Image Blurring").run()
    at.selectbox[0].set_value("Gaussian Blur").run()
    captions = [c.value for c in at.caption]
    assert any("Gaussian Blur" in c for c in captions)


# ── 9. Alternate kernel size ──────────────────────────────────────────────────

def test_kernel_size_9_no_exception() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Image Blurring").run()
    at.select_slider[0].set_value(9).run()
    assert not at.exception


def test_kernel_size_9_two_images() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Image Blurring").run()
    at.select_slider[0].set_value(9).run()
    if not hasattr(at, "image"):
        pytest.skip("image not tracked by this AppTest version")
    assert len(at.image) == 2


# ── 10. Public copy ───────────────────────────────────────────────────────────

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


# ── 11. Upload state contract ─────────────────────────────────────────────────

def _make_synthetic_png() -> bytes:
    """Return a deterministic 64x96 BGR PNG as bytes (seed=42, gradient+checkerboard)."""
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


@pytest.fixture(scope="module")
def at_upload() -> AppTest:
    """AppTest with Image Blurring page and synthetic image uploaded."""
    if not hasattr(AppTest.from_file(APP, default_timeout=60).run(), "file_uploader"):
        pytest.skip("file_uploader not tracked by this AppTest version")
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Image Blurring").run()
    at.file_uploader[0].set_value(_UPLOAD_TUPLE).run()
    return at


def test_upload_no_exception(at_upload: AppTest) -> None:
    assert not at_upload.exception


def test_upload_two_images(at_upload: AppTest) -> None:
    if not hasattr(at_upload, "image"):
        pytest.skip("image not tracked by this AppTest version")
    assert len(at_upload.image) == 2


def test_upload_subheaders_unchanged(at_upload: AppTest) -> None:
    sub_vals = [s.value for s in at_upload.subheader]
    assert sub_vals == EXPECTED_SUBHEADERS, f"Got {sub_vals}"


def test_upload_no_metrics(at_upload: AppTest) -> None:
    assert len(at_upload.metric) == 0


def test_upload_selectbox_default(at_upload: AppTest) -> None:
    assert at_upload.selectbox[0].value == "Average Blur"


def test_upload_slider_default(at_upload: AppTest) -> None:
    assert at_upload.select_slider[0].value == 5


def test_upload_kernel_caption_present(at_upload: AppTest) -> None:
    captions = [c.value for c in at_upload.caption]
    assert any("sum to" in c for c in captions)


def test_upload_border_caption_present(at_upload: AppTest) -> None:
    captions = [c.value for c in at_upload.caption]
    assert any("BORDER_REFLECT_101" in c for c in captions)
