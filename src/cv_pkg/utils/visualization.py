"""Visualization utilities for CV study."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence


def draw_bbox(
    image: np.ndarray,
    boxes: Sequence[tuple[int, int, int, int]],
    labels: Sequence[str] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image.

    Args:
        image: Input image (BGR or RGB).
        boxes: List of bounding boxes as (x1, y1, x2, y2).
        labels: Optional list of labels for each box.
        color: Box color in BGR format.
        thickness: Line thickness.

    Returns:
        Image with drawn bounding boxes.
    """
    img = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if labels and i < len(labels):
            cv2.putText(
                img, labels[i], (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
    return img


def show_images(
    images: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    cols: int = 3,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """Display multiple images in a grid.

    Args:
        images: List of images to display.
        titles: Optional list of titles for each image.
        cols: Number of columns in the grid.
        figsize: Figure size.
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, img in enumerate(images):
        if img.ndim == 3 and img.shape[2] == 3:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[i].imshow(img, cmap="gray")
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
