"""Contract tests for the Module 3 Experimental Validation page.

Covers:
1. Default (results available): header, subheaders, dataframe
2. Missing results: info message, no exception
3. Dataframe contract: 6 rows, correct columns, exact CSV values
4. Metric count unchanged (0)
5. No exceptions in either state
6. Status-message semantics (waiting state -> st.info, not st.warning)
7. Public copy: no em/en dashes, no internal path in visible text
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

APP = str(ROOT / "app.py")
CSV_PATH = ROOT / "results" / "experiment_results.csv"
MD_PATH = ROOT / "results" / "experiment_results.md"

# Expected values from the committed CSV (row 0)
EXPECTED_ROW0 = {
    "Experiment": 1,
    "Filter": "Average Blur",
    "Kernel Size": "3x3",
    "Spatial vs. Fourier MAE": pytest.approx(7.02476813698e-14, rel=1e-6),
    "MSE": pytest.approx(7.63627040472e-27, rel=1e-6),
    "RMSE": pytest.approx(8.73857563034e-14, rel=1e-6),
    "Max Abs Error": pytest.approx(3.41060513165e-13, rel=1e-6),
    "PSNR (dB)": pytest.approx(309.301990623, rel=1e-6),
    "Observation": "Required average blur baseline.",
}

EXPECTED_COLUMNS = [
    "Experiment", "Filter", "Kernel Size",
    "Spatial vs. Fourier MAE", "MSE", "RMSE", "Max Abs Error", "PSNR (dB)", "Observation",
]

EXPECTED_FILTERS = ["Average Blur", "Average Blur", "Gaussian Blur",
                    "Average Blur", "Gaussian Blur", "Gaussian Blur"]
EXPECTED_KERNEL_SIZES = ["3x3", "5x5", "5x5", "9x9", "3x3", "9x9"]


@pytest.fixture(scope="module")
def at_default() -> AppTest:
    """AppTest with the Experimental Validation page rendered (results present)."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Experimental Validation").run()
    return at


# ── 1. Default state: no exception ───────────────────────────────────────────

def test_default_no_exception(at_default: AppTest) -> None:
    assert not at_default.exception


# ── 2. Page structure ─────────────────────────────────────────────────────────

def test_default_header_value(at_default: AppTest) -> None:
    assert any(h.value == "Experimental Validation" for h in at_default.header)


def test_default_subheaders_present_and_ordered(at_default: AppTest) -> None:
    sub_vals = [s.value for s in at_default.subheader]
    assert "Input" in sub_vals
    assert "Validation Results" in sub_vals
    assert sub_vals.index("Input") < sub_vals.index("Validation Results")


def test_default_no_metrics(at_default: AppTest) -> None:
    """Migration must not add new st.metric elements."""
    assert len(at_default.metric) == 0


# ── 3. Dataframe contract ─────────────────────────────────────────────────────

def test_default_dataframe_present(at_default: AppTest) -> None:
    assert len(at_default.dataframe) == 1


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="results CSV not committed")
def test_dataframe_row_count(at_default: AppTest) -> None:
    df = at_default.dataframe[0].value
    assert df is not None
    assert len(df) == 6


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="results CSV not committed")
def test_dataframe_columns(at_default: AppTest) -> None:
    df = at_default.dataframe[0].value
    assert list(df.columns) == EXPECTED_COLUMNS


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="results CSV not committed")
def test_dataframe_row0_experiment_number(at_default: AppTest) -> None:
    df = at_default.dataframe[0].value
    assert int(df.iloc[0]["Experiment"]) == 1


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="results CSV not committed")
def test_dataframe_row0_filter(at_default: AppTest) -> None:
    df = at_default.dataframe[0].value
    assert df.iloc[0]["Filter"] == "Average Blur"


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="results CSV not committed")
def test_dataframe_row0_kernel_size(at_default: AppTest) -> None:
    df = at_default.dataframe[0].value
    assert df.iloc[0]["Kernel Size"] == "3x3"


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="results CSV not committed")
def test_dataframe_row0_mae(at_default: AppTest) -> None:
    df = at_default.dataframe[0].value
    assert float(df.iloc[0]["Spatial vs. Fourier MAE"]) == pytest.approx(7.02476813698e-14, rel=1e-6)


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="results CSV not committed")
def test_dataframe_row0_psnr(at_default: AppTest) -> None:
    df = at_default.dataframe[0].value
    assert float(df.iloc[0]["PSNR (dB)"]) == pytest.approx(309.301990623, rel=1e-6)


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="results CSV not committed")
def test_dataframe_row0_observation(at_default: AppTest) -> None:
    df = at_default.dataframe[0].value
    assert df.iloc[0]["Observation"] == "Required average blur baseline."


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="results CSV not committed")
def test_dataframe_all_filters(at_default: AppTest) -> None:
    df = at_default.dataframe[0].value
    assert list(df["Filter"]) == EXPECTED_FILTERS


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="results CSV not committed")
def test_dataframe_all_kernel_sizes(at_default: AppTest) -> None:
    df = at_default.dataframe[0].value
    assert list(df["Kernel Size"]) == EXPECTED_KERNEL_SIZES


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="results CSV not committed")
def test_dataframe_excludes_max_imaginary_column(at_default: AppTest) -> None:
    """max_imaginary_abs was not in the pre-migration Markdown table; it must not appear here."""
    df = at_default.dataframe[0].value
    assert "max_imaginary_abs" not in df.columns


# ── 4. Missing results state ──────────────────────────────────────────────────

def test_missing_results_shows_info_not_warning() -> None:
    """When experiment_results.md is absent the page must use st.info, not st.warning."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Patch the results dir to a temp dir with no files
        import module3.webapp.pages as _pages
        original = _pages._RESULTS_DIR
        _pages._RESULTS_DIR = tmp_path
        try:
            at = AppTest.from_file(APP, default_timeout=30).run()
            at.sidebar.radio[0].set_value("Experimental Validation").run()
            assert not at.exception
            assert len(at.info) >= 1, "missing-results state must use st.info"
            assert len(at.warning) == 0, "missing-results state must not use st.warning"
        finally:
            _pages._RESULTS_DIR = original


def test_missing_results_no_dataframe() -> None:
    """No dataframe is shown when experiment results are absent."""
    with tempfile.TemporaryDirectory() as tmp:
        import module3.webapp.pages as _pages
        original = _pages._RESULTS_DIR
        _pages._RESULTS_DIR = Path(tmp)
        try:
            at = AppTest.from_file(APP, default_timeout=30).run()
            at.sidebar.radio[0].set_value("Experimental Validation").run()
            assert len(at.dataframe) == 0
        finally:
            _pages._RESULTS_DIR = original


def test_missing_results_header_still_present() -> None:
    """The page header renders even when results are absent."""
    with tempfile.TemporaryDirectory() as tmp:
        import module3.webapp.pages as _pages
        original = _pages._RESULTS_DIR
        _pages._RESULTS_DIR = Path(tmp)
        try:
            at = AppTest.from_file(APP, default_timeout=30).run()
            at.sidebar.radio[0].set_value("Experimental Validation").run()
            assert any(h.value == "Experimental Validation" for h in at.header)
        finally:
            _pages._RESULTS_DIR = original


# ── 5. Public copy ────────────────────────────────────────────────────────────

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


def test_no_internal_path_reference(at_default: AppTest) -> None:
    text = _all_visible_text(at_default)
    # Internal filenames must not appear in visible user-facing text
    assert "module3_sample.png" not in text
    assert "experiment_results.md" not in text
    assert "experiment_results.csv" not in text


def test_no_phase_terminology(at_default: AppTest) -> None:
    text = _all_visible_text(at_default)
    for term in ["Phase 12", "Phase 11", "Phase 10", "migration", "AGENTS.md", "GIT_WORKFLOW"]:
        assert term not in text, f"Internal term {term!r} found in visible text"
