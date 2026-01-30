"""Data loading utilities for CV study."""

import cv2
import numpy as np
from pathlib import Path


def load_image(
    path: str | Path,
    color: bool = True,
) -> np.ndarray:
    """Load an image from file.

    Args:
        path: Path to the image file.
        color: If True, load as color image. Otherwise grayscale.

    Returns:
        Loaded image as numpy array (BGR if color).

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If image cannot be loaded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def preprocess_image(
    image: np.ndarray,
    size: tuple[int, int] = (640, 640),
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess image for model input.

    Args:
        image: Input image (BGR).
        size: Target size as (width, height).
        normalize: If True, normalize pixel values to [0, 1].

    Returns:
        Preprocessed image.
    """
    img = cv2.resize(image, size)
    if normalize:
        img = img.astype(np.float32) / 255.0
    return img
