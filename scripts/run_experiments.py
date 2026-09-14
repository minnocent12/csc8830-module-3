"""Run the Module 3 spatial-vs-Fourier validation matrix.

Run from the Module_3 repository root:

    python scripts/run_experiments.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from module3.comparison import compare_spatial_fourier  # noqa: E402
from module3.experiments import (  # noqa: E402
    EXPERIMENT_MATRIX,
    kernel_for_spec,
    rows_to_csv,
    rows_to_markdown,
    run_experiments_for_image_path,
)
from module3.io_utils import bgr_to_gray_float64, load_image_bgr  # noqa: E402
from module3.visualization import save_comparison_figure, save_kernel_figure  # noqa: E402

DEFAULT_IMAGE = REPO_ROOT / "data" / "sample_images" / "module3_sample.png"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="input image path")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "docs" / "report" / "figures")
    return parser.parse_args()


def main() -> None:
    """Run experiments, write result tables, and save figures."""
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(
            f"Missing image: {args.image}\n"
            "Create the default with: python scripts/generate_sample_image.py"
        )

    rows = run_experiments_for_image_path(args.image)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "experiment_results.csv").write_text(rows_to_csv(rows), encoding="utf-8")
    (args.results_dir / "experiment_results.md").write_text(
        rows_to_markdown(rows, image_name=args.image.name),
        encoding="utf-8",
    )

    bgr = load_image_bgr(args.image)
    gray = bgr_to_gray_float64(bgr)
    for spec in EXPERIMENT_MATRIX:
        kernel_spec = kernel_for_spec(spec)
        result = compare_spatial_fourier(gray, kernel_spec.kernel)
        stem = f"experiment_{spec.experiment}_{spec.filter_key}_{spec.kernel_size}x{spec.kernel_size}"
        save_comparison_figure(
            result,
            args.figures_dir / f"{stem}_comparison.png",
            title=f"{kernel_spec.filter_name} {kernel_spec.size}x{kernel_spec.size}",
        )
        save_kernel_figure(
            kernel_spec.kernel,
            args.figures_dir / f"{stem}_kernel.png",
            title=f"{kernel_spec.filter_name} kernel",
        )

    print(f"Wrote {args.results_dir / 'experiment_results.csv'}")
    print(f"Wrote {args.results_dir / 'experiment_results.md'}")
    print(f"Wrote figures under {args.figures_dir}")
    for row in rows:
        print(
            f"{row.experiment}: {row.filter_name} {row.kernel_size}x{row.kernel_size} "
            f"MAE={row.mae:.3e} MSE={row.mse:.3e}"
        )


if __name__ == "__main__":
    main()
