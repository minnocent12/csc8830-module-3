from __future__ import annotations

import pytest

from module3.webapp._page import PageSpec
from module3.webapp.pages import get_pages
from module3.webapp.registry import collect_pages


def _fake_module_provider() -> list[PageSpec]:
    return [
        PageSpec("Module X", "Overview", 10, lambda: None),
        PageSpec("Module X", "Details", 20, lambda: None),
    ]


def test_single_provider_returns_module3_pages_in_order() -> None:
    pages = collect_pages([get_pages])
    assert [p.page_label for p in pages] == [
        "Image Blurring",
        "Spatial vs Fourier",
        "Experimental Validation",
        "Theory",
    ]
    assert {p.module_label for p in pages} == {"Module 3"}


def test_multiple_providers_merge_grouped_and_ordered() -> None:
    pages = collect_pages([get_pages, _fake_module_provider])
    assert [(p.module_label, p.page_label) for p in pages] == [
        ("Module 3", "Image Blurring"),
        ("Module 3", "Spatial vs Fourier"),
        ("Module 3", "Experimental Validation"),
        ("Module 3", "Theory"),
        ("Module X", "Overview"),
        ("Module X", "Details"),
    ]


def test_later_provider_overrides_same_key() -> None:
    marker = object()

    def _override() -> list[PageSpec]:
        return [PageSpec("Module 3", "Theory", 40, lambda: marker)]

    pages = collect_pages([get_pages, _override])
    theory = next(p for p in pages if p.page_label == "Theory")
    assert theory.render() is marker


def test_non_pagespec_is_rejected() -> None:
    with pytest.raises(TypeError):
        collect_pages([lambda: ["not a page"]])
