"""Debug visualizations for the segmentation stage."""

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.pipeline.debug import DebugContext


def create_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Create image with semi-transparent mask overlay.

    Args:
        image: Original BGR image
        mask: Binary mask (0 or 255)
        color: Overlay color (BGR)
        alpha: Overlay transparency

    Returns:
        Image with overlay
    """
    overlay = image.copy()
    mask_bool = mask > 127

    # Create colored overlay
    colored = np.zeros_like(image)
    colored[mask_bool] = color

    # Blend
    overlay = cv2.addWeighted(image, 1.0, colored, alpha, 0)

    # Draw contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay


def create_click_point_visualization(
    image: np.ndarray,
    click_x: float,
    click_y: float,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Create visualization showing click point and resulting mask.

    Args:
        image: Original image
        click_x: Normalized x coordinate (0-1)
        click_y: Normalized y coordinate (0-1)
        mask: Resulting segmentation mask

    Returns:
        Visualization image
    """
    h, w = image.shape[:2]
    px, py = int(click_x * w), int(click_y * h)

    # Create overlay
    vis = create_mask_overlay(image, mask, (0, 255, 0), 0.3)

    # Draw click point
    cv2.circle(vis, (px, py), 15, (0, 0, 255), 3)
    cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)
    cv2.line(vis, (px - 25, py), (px + 25, py), (0, 0, 255), 2)
    cv2.line(vis, (px, py - 25), (px, py + 25), (0, 0, 255), 2)

    # Add label
    cv2.putText(
        vis, f"Click: ({px}, {py})",
        (px + 20, py - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )

    return vis


def create_mask_comparison(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    max_cols: int = 4,
    thumb_size: int = 300,
) -> np.ndarray:
    """
    Create grid showing all masks side by side.

    Args:
        images: Original images
        masks: Segmentation masks
        max_cols: Maximum columns
        thumb_size: Thumbnail size

    Returns:
        Comparison grid image
    """
    n_images = len(images)
    n_cols = min(n_images, max_cols)
    n_rows = (n_images + n_cols - 1) // n_cols

    thumbnails = []
    for i, (img, mask) in enumerate(zip(images, masks)):
        # Create overlay
        overlay = create_mask_overlay(img, mask, (0, 255, 0), 0.4)

        # Resize
        h, w = overlay.shape[:2]
        scale = thumb_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        thumb = cv2.resize(overlay, (new_w, new_h))

        # Pad to square
        pad_h = thumb_size - new_h
        pad_w = thumb_size - new_w
        thumb = cv2.copyMakeBorder(
            thumb, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(40, 40, 40)
        )

        # Add label
        mask_pct = (mask > 127).sum() / mask.size * 100
        cv2.putText(
            thumb, f"[{i}] {mask_pct:.1f}%",
            (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        thumbnails.append(thumb)

    # Pad to fill grid
    while len(thumbnails) < n_rows * n_cols:
        thumbnails.append(np.full((thumb_size, thumb_size, 3), 40, dtype=np.uint8))

    # Arrange in grid
    rows = []
    for r in range(n_rows):
        row_imgs = thumbnails[r * n_cols:(r + 1) * n_cols]
        rows.append(np.hstack(row_imgs))
    comparison = np.vstack(rows)

    return comparison


def compute_mask_stats(mask: np.ndarray) -> dict:
    """Compute statistics for a segmentation mask."""
    h, w = mask.shape[:2]
    mask_bool = mask > 127

    # Basic stats
    area_pixels = mask_bool.sum()
    area_pct = area_pixels / mask_bool.size * 100

    # Bounding box
    if area_pixels > 0:
        coords = np.where(mask_bool)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        bbox_area = (x_max - x_min) * (y_max - y_min)
        fill_ratio = area_pixels / bbox_area if bbox_area > 0 else 0
    else:
        bbox = [0, 0, 0, 0]
        fill_ratio = 0

    # Connectivity (number of connected components)
    num_labels, labels = cv2.connectedComponents(mask)
    num_components = num_labels - 1  # Subtract background

    return {
        "area_pixels": int(area_pixels),
        "area_percent": float(area_pct),
        "bounding_box": bbox,
        "fill_ratio": float(fill_ratio),
        "num_components": int(num_components),
    }


def debug_segmentation(
    ctx: "DebugContext",
    images: list[np.ndarray],
    masks: list[np.ndarray],
    click_point: tuple[float, float],
    reference_idx: int,
) -> None:
    """
    Generate debug visualizations for the segmentation stage.

    Outputs:
        - mask_overlay_XXX.jpg: Each image with mask overlay
        - click_point.jpg: Reference image with click point marked
        - mask_comparison.jpg: Grid showing all masks
        - mask_stats.json: Per-image mask statistics

    Args:
        ctx: Debug context
        images: Original images
        masks: Segmentation masks
        click_point: (x, y) normalized click coordinates
        reference_idx: Index of reference image
    """
    if not ctx.enabled:
        return

    # Save individual mask overlays
    for i, (img, mask) in enumerate(zip(images, masks)):
        overlay = create_mask_overlay(img, mask)
        ctx.save_image("segmentation", f"mask_overlay_{i:03d}", overlay)

    # Save click point visualization
    click_vis = create_click_point_visualization(
        images[reference_idx],
        click_point[0],
        click_point[1],
        masks[reference_idx],
    )
    ctx.save_image("segmentation", "click_point", click_vis)

    # Save mask comparison grid
    comparison = create_mask_comparison(images, masks)
    ctx.save_image("segmentation", "mask_comparison", comparison)

    # Compute and save stats
    stats = {
        "reference_image_index": reference_idx,
        "click_point": {"x": click_point[0], "y": click_point[1]},
        "num_images": len(images),
        "per_image_stats": [
            {"index": i, **compute_mask_stats(mask)}
            for i, mask in enumerate(masks)
        ],
    }
    ctx.save_json("segmentation", "mask_stats", stats)
