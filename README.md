# CSc 8830 Module 3 - Spatial and Fourier-Domain Image Blurring

**Public web app (all modules):** <https://csc8830-dashboard-minnocent1.streamlit.app>

This repository contains the Module 3 assignment for Georgia State University CSc 8830:
Computer Vision. The project demonstrates image blurring in the spatial domain and verifies,
mathematically and experimentally, that the equivalent Fourier-domain operation produces the
same result when padding, kernel alignment, and boundary handling are matched correctly.

The implementation is written in Python using OpenCV, NumPy, Matplotlib, pytest, and
Streamlit.

## What This Project Demonstrates

- Spatial-domain image blurring with explicit average and Gaussian kernels.
- Fourier-domain filtering with `fft2`, frequency-domain multiplication, and `ifft2`.
- A typed convolution-theorem derivation in `docs/CONVOLUTION_THEOREM.md`.
- Numerical validation using MAE, MSE, RMSE, maximum absolute error, and PSNR.
- Difference-image visualization between spatial and Fourier results.
- A standalone Streamlit web application exposing the required assignment components.
- Reproducible sample data, result tables, figures, and tests.

## Repository Structure

```text
.
├── app.py                         # Streamlit entry point
├── data/sample_images/            # Deterministic bundled sample image
├── docs/                          # Theory, implementation notes, results discussion
├── docs/report/figures/           # Generated comparison and kernel figures
├── results/                       # Generated experiment CSV and Markdown table
├── scripts/
│   ├── generate_sample_image.py    # Creates the deterministic sample input
│   └── run_experiments.py          # Recomputes validation metrics and figures
├── src/module3/                   # Importable core package and web app pages
└── tests/                         # Synthetic pytest coverage
```

## Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/minnocent12/csc8830-module-3.git
cd csc8830-module-3
```

Then create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run the Web Application

```bash
streamlit run app.py
```

The app includes four pages:

- `Image Blurring`: upload/select an image, choose a blur filter, and preview the spatial
  result.
- `Spatial vs Fourier`: compare matched spatial and Fourier filtering on grayscale data.
- `Experimental Validation`: view the generated MAE/MSE results table.
- `Theory`: read the typed convolution-theorem derivation.

## Optional: Run Multiple Modules in One Dashboard

This repository is independently runnable and gradable by itself. If multiple module
repositories are available, they can also be placed beside each other and mounted in one
Streamlit dashboard with a module dropdown.

Create any parent workspace directory and clone the module repositories into it:

```bash
mkdir csc8830-workspace
cd csc8830-workspace
git clone https://github.com/minnocent12/csc8830-module-2.git
git clone https://github.com/minnocent12/csc8830-module-3.git
```

The directory should look like this:

```text
csc8830-workspace/
├── csc8830-module-2/
└── csc8830-module-3/
```

Create a root dashboard file named `app.py` in `csc8830-workspace/`:

```python
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
for src in ROOT.glob("csc8830-module-*/src"):
    sys.path.insert(0, str(src))

from module2.webapp.pages import get_pages as get_module2_pages
from module3.webapp.pages import get_pages as get_module3_pages


def main() -> None:
    st.set_page_config(page_title="CSc 8830 Computer Vision", layout="wide")
    pages = [*get_module2_pages(), *get_module3_pages()]

    by_module = defaultdict(list)
    for page in pages:
        by_module[page.module_label].append(page)

    with st.sidebar:
        st.title("CSc 8830 Computer Vision")
        module_label = st.selectbox("Module", list(by_module))
        module_pages = sorted(by_module[module_label], key=lambda page: page.order)
        page_label = st.radio("Page", [page.page_label for page in module_pages])

    selected = next(page for page in module_pages if page.page_label == page_label)
    selected.render()


if __name__ == "__main__":
    main()
```

Install the modules into one environment and run the dashboard from the parent workspace:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e csc8830-module-2
python -m pip install -e csc8830-module-3
streamlit run app.py
```

Module 3 supports this shared-dashboard workflow by exposing its pages through:

```python
from module3.webapp.pages import get_pages
```

The standalone Module 3 command remains the recommended grading path for this repository:

```bash
streamlit run app.py
```

## Reproduce the Experiment Results

Generate the deterministic sample image:

```bash
python scripts/generate_sample_image.py
```

Run the full validation matrix:

```bash
python scripts/run_experiments.py
```

This writes:

- `data/sample_images/module3_sample.png`
- `results/experiment_results.csv`
- `results/experiment_results.md`
- comparison and kernel figures in `docs/report/figures/`

The generated table compares average and Gaussian filters across multiple kernel sizes. The
measured errors are near machine precision because both implementations use the same grayscale
`float64` input, normalized kernel, zero-padding strategy, and boundary model.

## Run Tests

```bash
python -m pytest -q
```

The tests cover kernel generation, image I/O, spatial filtering, Fourier alignment, comparison
metrics, experiment generation, and the Streamlit page-registration contract.

## Implementation Summary

The canonical validation path operates on single-channel grayscale `float64` arrays. Spatial
filtering uses `cv2.filter2D` with `cv2.BORDER_CONSTANT`. The Fourier path zero-pads the image
by the kernel radius, embeds the kernel in the padded array, rolls the kernel center to the FFT
origin, multiplies spectra, applies the inverse FFT, and crops back to the original image size.

The display-only filtering demo applies the same kernel independently to each BGR channel with
`BORDER_REFLECT_101`, which avoids visually distracting dark borders on uploaded photos. That
preview path is intentionally separate from the numerical spatial-vs-Fourier validation.

## Documentation

- `docs/CONVOLUTION_THEOREM.md`: typed mathematical derivation.
- `docs/IMPLEMENTATION_NOTES.md`: padding, alignment, boundary, precision, and color handling.
- `docs/EXPERIMENTAL_RESULTS.md`: discussion of measured results.
- `docs/architecture.md`: project organization and module boundaries.

## Current Validation Status

- Implemented: core algorithms, experiment scripts, Streamlit app, tests, and documentation.
- Tested: automated pytest suite passes.
- Experimentally validated: results table generated by `scripts/run_experiments.py`.
- Pending for final course submission: instructor access to the private GitHub repository,
  demonstration video recording, final PDF assembly, and Google Classroom upload.
