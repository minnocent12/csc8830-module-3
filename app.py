"""Standalone Streamlit entry point for the CSc 8830 Module 3 submission app.

Run it from this repository root:

    streamlit run app.py

or from the Assignments workspace root:

    streamlit run Module_3/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from module3.webapp.pages import get_pages  # noqa: E402
from module3.webapp.registry import collect_pages  # noqa: E402
from module3.webapp.shell import render_app  # noqa: E402


def main() -> None:
    """Collect Module 3 pages and render the standalone submission app."""
    render_app(collect_pages([get_pages]))


if __name__ == "__main__":
    main()
