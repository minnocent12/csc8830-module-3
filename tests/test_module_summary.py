"""The optional Home summary reports status only from committed evidence."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from module3.webapp.summary import RESULTS_AVAILABLE, get_module_summary
from module3.webapp.summary import RESULTS_TABLE

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (RESULTS_TABLE,)


def copy_evidence(dest: Path) -> Path:
    for rel in EVIDENCE:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dest / rel)
    return dest


def test_committed_evidence_reports_results_available() -> None:
    assert get_module_summary() == RESULTS_AVAILABLE
    assert (RESULTS_AVAILABLE.label, RESULTS_AVAILABLE.kind) == ("Results available", "success")


def test_evidence_files_are_committed() -> None:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", *EVIDENCE], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    assert sorted(out) == sorted(EVIDENCE)


def test_copied_evidence_gives_the_same_status(tmp_path: Path) -> None:
    assert get_module_summary(copy_evidence(tmp_path)) == RESULTS_AVAILABLE


@pytest.mark.parametrize("missing", EVIDENCE)
def test_any_missing_evidence_gives_no_status(tmp_path: Path, missing: str) -> None:
    copy_evidence(tmp_path)
    (tmp_path / missing).unlink()
    assert get_module_summary(tmp_path) is None


def test_no_evidence_gives_no_status(tmp_path: Path) -> None:
    assert get_module_summary(tmp_path) is None


def test_reading_the_summary_writes_nothing(tmp_path: Path) -> None:
    copy_evidence(tmp_path)
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()}
    get_module_summary(tmp_path)
    after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_the_summary_does_not_load_opencv() -> None:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import module3.webapp.summary as s; s.get_module_summary(); print('cv2' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(REPO_ROOT / "src")], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_a_header_only_table_gives_no_status(tmp_path: Path) -> None:
    copy_evidence(tmp_path)
    header = (tmp_path / RESULTS_TABLE).read_text(encoding="utf-8").splitlines()[0]
    (tmp_path / RESULTS_TABLE).write_text(header + "\n", encoding="utf-8")
    assert get_module_summary(tmp_path) is None


def test_a_table_missing_required_columns_gives_no_status(tmp_path: Path) -> None:
    copy_evidence(tmp_path)
    (tmp_path / RESULTS_TABLE).write_text("experiment,filter\n1,Average Blur\n", encoding="utf-8")
    assert get_module_summary(tmp_path) is None
