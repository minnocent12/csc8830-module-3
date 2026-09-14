from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from module3.io_utils import (
    bgr_to_gray_float64,
    decode_image_bgr,
    ensure_gray_float64,
    load_image_bgr,
    save_image,
    to_display_uint8,
)


def test_load_decode_and_gray_conversion_round_trip(tmp_path: Path) -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 80
    image[:, :, 2] = 220
    path = tmp_path / "image.png"
    save_image(path, image)

    loaded = load_image_bgr(path)
    assert loaded.shape == image.shape

    ok, encoded = cv2.imencode(".png", image)
    assert ok
    decoded = decode_image_bgr(encoded.tobytes())
    assert decoded.shape == image.shape

    gray = bgr_to_gray_float64(decoded)
    assert gray.dtype == np.float64
    assert gray.shape == image.shape[:2]


def test_invalid_image_inputs_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image_bgr(tmp_path / "missing.png")
    with pytest.raises(ValueError):
        decode_image_bgr(b"not an image")
    with pytest.raises(ValueError):
        bgr_to_gray_float64(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(ValueError):
        ensure_gray_float64(np.zeros((5, 5, 3), dtype=np.uint8))


def test_to_display_uint8_clips_and_rounds() -> None:
    image = np.array([[-5.0, 0.49, 1.5, 300.0]])
    assert to_display_uint8(image).tolist() == [[0, 0, 2, 255]]
