"""Debug visualizations for the preprocessing stage."""

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.pipeline.debug import DebugContext
    from app.pipeline.preprocess import CameraIntrinsics


def create_image_gallery(
    images: list[np.ndarray],
    image_paths: list[Path],
    max_cols: int = 4,
    thumb_size: int = 400,
) -> np.ndarray:
    """
    Create a gallery grid of all input images with labels.

    Args:
        images: List of preprocessed images
        image_paths: Original file paths
        max_cols: Maximum columns in gallery
        thumb_size: Thumbnail size (longest edge)

    Returns:
        Gallery image
    """
    n_images = len(images)
    n_cols = min(n_images, max_cols)
    n_rows = (n_images + n_cols - 1) // n_cols

    thumbnails = []
    for i, (img, path) in enumerate(zip(images, image_paths)):
        # Resize to thumbnail
        h, w = img.shape[:2]
        scale = thumb_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        thumb = cv2.resize(img, (new_w, new_h))

        # Pad to square
        pad_h = thumb_size - new_h
        pad_w = thumb_size - new_w
        thumb = cv2.copyMakeBorder(
            thumb,
            0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT,
            value=(40, 40, 40)
        )

        # Add label
        label = f"[{i}] {path.name}"
        dims = f"{w}x{h}"
        cv2.putText(thumb, label[:30], (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(thumb, dims, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        thumbnails.append(thumb)

    # Pad to fill grid
    while len(thumbnails) < n_rows * n_cols:
        thumbnails.append(np.full((thumb_size, thumb_size, 3), 40, dtype=np.uint8))

    # Arrange in grid
    rows = []
    for r in range(n_rows):
        row_imgs = thumbnails[r * n_cols:(r + 1) * n_cols]
        rows.append(np.hstack(row_imgs))
    gallery = np.vstack(rows)

    return gallery


def debug_preprocessing(
    ctx: "DebugContext",
    images: list[np.ndarray],
    intrinsics: "CameraIntrinsics",
    image_paths: list[Path],
) -> None:
    """
    Generate debug visualizations for the preprocessing stage.

    Outputs:
        - image_gallery.jpg: Grid of all input images with dimensions
        - intrinsics.json: Camera intrinsics data
        - preprocessing_summary.json: Per-image processing info

    Args:
        ctx: Debug context
        images: Preprocessed images
        intrinsics: Extracted/estimated camera intrinsics
        image_paths: Original image paths
    """
    if not ctx.enabled:
        return

    # Create image gallery
    gallery = create_image_gallery(images, image_paths)
    ctx.save_image("preprocess", "image_gallery", gallery)

    # Save intrinsics
    intrinsics_data = {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "width": intrinsics.width,
        "height": intrinsics.height,
        "source": "exif" if hasattr(intrinsics, "_from_exif") else "estimated",
        "camera_matrix": intrinsics.to_matrix().tolist(),
    }
    ctx.save_json("preprocess", "intrinsics", intrinsics_data)

    # Save preprocessing summary
    summary = {
        "num_images": len(images),
        "processed_dimensions": {
            "width": images[0].shape[1] if images else 0,
            "height": images[0].shape[0] if images else 0,
        },
        "images": [
            {
                "index": i,
                "filename": path.name,
                "original_path": str(path),
                "processed_shape": list(img.shape),
            }
            for i, (img, path) in enumerate(zip(images, image_paths))
        ],
    }
    ctx.save_json("preprocess", "preprocessing_summary", summary)
