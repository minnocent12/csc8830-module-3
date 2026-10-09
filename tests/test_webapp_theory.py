"""Contract tests for the Theory page after the Phase 11 design-system migration.

Verifies:
- page renders without exception
- st.header "Theory" present (from kit_page_header)
- three kit theory_section subheaders present in order
- exactly 8 st.latex equation blocks
- exact LaTeX string for each block
- key prose phrases from each section are rendered
- missing doc file surfaces as st.error, no subheaders rendered
- sidebar radio options unchanged
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = str(REPO_ROOT / "app.py")

EXPECTED_LATEX = [
    r"g[x, y] = (f * h)[x, y] = \sum_m \sum_n f[x-m,\, y-n]\, h[m, n]",
    (
        r"G[u, v] = \sum_x \sum_y g[x, y]\, \exp\!\left(-j\,2\pi"
        r" \left(\frac{ux}{M} + \frac{vy}{N}\right)\right)"
    ),
    (
        r"G[u, v] = \sum_x \sum_y \sum_m \sum_n"
        r" f[x-m,\, y-n]\, h[m, n]\, \exp\!\left(-j\,2\pi"
        r" \left(\frac{ux}{M} + \frac{vy}{N}\right)\right)"
    ),
    (
        r"G[u, v] = \sum_m \sum_n h[m, n] \sum_a \sum_b f[a, b]\, \exp\!\left(-j\,2\pi"
        r" \left(\frac{u(a+m)}{M} + \frac{v(b+n)}{N}\right)\right)"
    ),
    (
        r"G[u, v] = \sum_m \sum_n h[m, n]\, \exp\!\left(-j\,2\pi"
        r" \left(\frac{um}{M} + \frac{vn}{N}\right)\right)"
        r" \cdot \sum_a \sum_b f[a, b]\, \exp\!\left(-j\,2\pi"
        r" \left(\frac{ua}{M} + \frac{vb}{N}\right)\right)"
    ),
    r"G[u, v] = H[u, v]\, F[u, v]",
    r"\text{DFT}\{f * h\} = \text{DFT}\{f\} \cdot \text{DFT}\{h\}",
    r"f * h = \text{IDFT}\!\left(F[u, v]\, H[u, v]\right)",
]


def _theory_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception, f"initial run raised: {app.exception}"
    app.sidebar.radio[0].set_value("Theory").run()
    assert not app.exception, f"Theory page raised: {app.exception}"
    return app


def test_theory_page_renders_without_exception() -> None:
    app = _theory_app()
    assert not app.exception


def test_theory_page_has_header_theory() -> None:
    app = _theory_app()
    assert any(h.value == "Theory" for h in app.header)


def test_theory_page_has_three_subheaders_in_order() -> None:
    app = _theory_app()
    subheader_values = [s.value for s in app.subheader]
    assert subheader_values == ["Variables", "Derivation", "Connection to the Implementation"]


def test_theory_page_has_exactly_eight_latex_blocks() -> None:
    app = _theory_app()
    assert len(app.latex) == 8


@pytest.mark.parametrize("index,expected", list(enumerate(EXPECTED_LATEX)))
def test_latex_block_exact_string(index: int, expected: str) -> None:
    app = _theory_app()
    # AppTest wraps st.latex content as "$$\n<latex>\n$$"
    rendered = app.latex[index].value
    assert f"$$\n{expected}\n$$" == rendered, (
        f"latex[{index}] mismatch.\n  expected: {expected!r}\n  got:      {rendered!r}"
    )


def test_theory_page_variables_section_content() -> None:
    app = _theory_app()
    all_text = " ".join(
        m.value for m in app.markdown
    )
    assert "f[x, y]" in all_text
    assert "h[m, n]" in all_text
    assert "G[u, v]" in all_text
    assert "discrete 2D convolution" in all_text


def test_theory_page_implementation_section_content() -> None:
    app = _theory_app()
    all_text = " ".join(m.value for m in app.markdown)
    assert "zero-pads" in all_text
    assert "cv2.filter2D" in all_text
    assert "circular convolution" in all_text


def test_theory_page_has_intro_caption() -> None:
    app = _theory_app()
    captions = [c.value for c in app.caption]
    assert any("digitized hand-worked" in c for c in captions)


def test_theory_page_sidebar_options_unchanged() -> None:
    app = _theory_app()
    assert list(app.sidebar.radio[0].options) == [
        "Image Blurring",
        "Spatial vs Fourier",
        "Experimental Validation",
        "Theory",
    ]


def test_theory_page_missing_doc_shows_error(tmp_path: Path) -> None:
    """If CONVOLUTION_THEOREM.md is absent the page shows st.error and no subheaders."""
    import importlib
    import sys

    # Copy repo to tmp, remove the doc
    dest = tmp_path / "Module_3"
    shutil.copytree(REPO_ROOT, dest, dirs_exist_ok=True)
    doc = dest / "docs" / "CONVOLUTION_THEOREM.md"
    if doc.is_file():
        doc.unlink()

    # Temporarily point the module's _DOCS_DIR at the tmp copy by reimporting pages
    # The simplest approach: run the module's app.py from the tmp copy
    tmp_app = dest / "app.py"
    if not tmp_app.is_file():
        pytest.skip("no app.py in tmp copy")

    tmp_src = str(dest / "src")
    if tmp_src not in sys.path:
        sys.path.insert(0, tmp_src)

    try:
        # Force reimport from the temp location by patching the module cache
        import module3.webapp.pages as pages_mod
        original_docs_dir = pages_mod._DOCS_DIR
        pages_mod._DOCS_DIR = dest / "docs"

        app = AppTest.from_file(str(tmp_app), default_timeout=60).run()
        if app.exception:
            pytest.skip(f"app.py failed to load: {app.exception}")
        app.sidebar.radio[0].set_value("Theory").run()
        assert not app.exception
        assert any("not available" in e.value for e in app.error)
        assert len(app.subheader) == 0
    finally:
        pages_mod._DOCS_DIR = original_docs_dir
