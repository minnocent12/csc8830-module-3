# CSc 8830 Module 3 - Spatial vs Fourier Image Blurring

This repository implements the Module 3 computer vision assignment: spatial-domain blur
filtering, equivalent Fourier-domain filtering, numerical validation, and a Streamlit web
application that exposes all major components.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Reproduce Results

Generate the deterministic sample image:

```bash
python scripts/generate_sample_image.py
```

Run the validation matrix and save result tables and figures:

```bash
python scripts/run_experiments.py
```

Run automated tests:

```bash
python -m pytest -q
```

## Run the Web App

```bash
streamlit run app.py
```

The app includes filtering controls, spatial/Fourier comparison, generated experiment results,
and the typed convolution-theorem derivation.

## Implementation Notes

The numerical comparison pipeline uses grayscale `float64` images, explicit normalized kernels,
matched zero boundary conditions, and a kernel-origin alignment step before FFT. The display
demo applies the same kernel independently to each BGR channel with `BORDER_REFLECT_101` so
uploaded photos look natural at the edges.

## Known Limitations

Generated experiment tables and figures are intentionally reproducible artifacts. If they are
missing, run `scripts/generate_sample_image.py` and `scripts/run_experiments.py`. The final PDF,
GitHub repository link, and demonstration video still need to be prepared for submission.
