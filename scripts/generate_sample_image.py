"""Generate the deterministic sample image used by Module 3.

Run from the Module_3 repository root:

    python scripts/generate_sample_image.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from module3.io_utils import save_image  # noqa: E402


def make_sample_image(width: int = 384, height: int = 256) -> np.ndarray:
    """Create a deterministic BGR image with gradients, edges, and shapes."""
    y = np.linspace(0, 1, height, dtype=np.float64)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float64)[None, :]

    blue = 255 * x
    green = 255 * y
    red = 120 + 80 * np.sin(2 * np.pi * (3 * x + 2 * y))
    image = np.dstack(
        [
            np.broadcast_to(blue, (height, width)),
            np.broadcast_to(green, (height, width)),
            red,
        ]
    )

    checker = ((np.floor(x * 16) + np.floor(y * 12)) % 2) * 42
    image[:, :, 1] = np.clip(image[:, :, 1] + checker, 0, 255)

    bgr = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    cv2.rectangle(bgr, (28, 30), (142, 146), (20, 235, 240), thickness=-1)
    cv2.circle(bgr, (284, 78), 44, (240, 60, 40), thickness=-1)
    cv2.line(bgr, (30, 212), (350, 180), (255, 255, 255), thickness=5)
    cv2.putText(bgr, "FFT", (210, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (10, 10, 10), 4)
    cv2.putText(bgr, "FFT", (210, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (250, 250, 250), 2)
    return bgr


def main() -> None:
    """Write the sample image to ``data/sample_images/module3_sample.png``."""
    output = REPO_ROOT / "data" / "sample_images" / "module3_sample.png"
    save_image(output, make_sample_image())
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
