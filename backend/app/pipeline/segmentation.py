import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from app.models import ClickPoint


# Global SAM model cache
_sam_model = None
_sam_predictor = None


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sam_model():
    """Load SAM model (cached globally)."""
    global _sam_model, _sam_predictor

    if _sam_predictor is not None:
        return _sam_predictor

    from segment_anything import sam_model_registry, SamPredictor

    device = get_device()

    # Try to find SAM checkpoint
    model_type = "vit_h"
    checkpoint_paths = [
        Path("/models/sam_vit_h.pth"),
        Path("models/sam_vit_h.pth"),
        Path.home() / ".cache/sam/sam_vit_h.pth",
    ]

    checkpoint_path = None
    for path in checkpoint_paths:
        if path.exists():
            checkpoint_path = path
            break

    if checkpoint_path is None:
        # Fall back to smaller model
        model_type = "vit_b"
        fallback_paths = [
            Path("/models/sam_vit_b.pth"),
            Path("models/sam_vit_b.pth"),
            Path.home() / ".cache/sam/sam_vit_b.pth",
        ]
        for path in fallback_paths:
            if path.exists():
                checkpoint_path = path
                break

    if checkpoint_path is None:
        raise RuntimeError(
            "SAM model checkpoint not found. Please download from "
            "https://github.com/facebookresearch/segment-anything#model-checkpoints"
        )

    _sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    _sam_model.to(device)
    _sam_predictor = SamPredictor(_sam_model)

    return _sam_predictor


def segment_single_image(
    image: np.ndarray,
    click_point: tuple[float, float],
) -> np.ndarray:
    """
    Segment the surfboard in a single image using SAM.

    Args:
        image: BGR image
        click_point: Normalized (x, y) coordinates where user clicked

    Returns:
        Binary mask where surfboard is 255, background is 0
    """
    predictor = load_sam_model()

    # Convert BGR to RGB for SAM
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set the image
    predictor.set_image(image_rgb)

    # Convert normalized coordinates to pixel coordinates
    h, w = image.shape[:2]
    point_x = int(click_point[0] * w)
    point_y = int(click_point[1] * h)

    # Predict mask using point prompt
    input_point = np.array([[point_x, point_y]])
    input_label = np.array([1])  # 1 = foreground

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Select the mask with highest score
    best_idx = np.argmax(scores)
    mask = masks[best_idx]

    # Convert to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    return mask_uint8


def propagate_mask_to_other_images(
    reference_image: np.ndarray,
    reference_mask: np.ndarray,
    target_images: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Propagate segmentation from reference image to other images.

    Uses feature matching to find corresponding regions in other views.
    Falls back to SAM with estimated click points if matching fails.
    """
    predictor = load_sam_model()
    masks = []

    # Find centroid of reference mask to use as approximate click point
    moments = cv2.moments(reference_mask)
    if moments["m00"] > 0:
        ref_cx = moments["m10"] / moments["m00"]
        ref_cy = moments["m01"] / moments["m00"]
    else:
        h, w = reference_mask.shape[:2]
        ref_cx, ref_cy = w / 2, h / 2

    # Normalized reference centroid
    h, w = reference_image.shape[:2]
    ref_norm_x = ref_cx / w
    ref_norm_y = ref_cy / h

    for target_image in target_images:
        # Use SAM with estimated click point
        # In a more sophisticated system, we'd use feature matching
        # to find the corresponding point
        target_h, target_w = target_image.shape[:2]

        # For now, use same normalized position
        # This works reasonably well if the board is roughly centered
        mask = segment_single_image(
            target_image,
            (ref_norm_x, ref_norm_y),
        )
        masks.append(mask)

    return masks


async def segment_board(
    images: list[np.ndarray],
    click_point: ClickPoint,
) -> list[np.ndarray]:
    """
    Segment the surfboard in all images.

    Uses user click point on the specified image, then propagates
    the segmentation to other views.

    Args:
        images: List of BGR images
        click_point: User click point specifying which image and where

    Returns:
        List of binary masks for each image
    """
    loop = asyncio.get_event_loop()

    # Segment the reference image (where user clicked)
    reference_idx = click_point.image_index
    reference_image = images[reference_idx]

    reference_mask = await loop.run_in_executor(
        None,
        segment_single_image,
        reference_image,
        (click_point.x, click_point.y),
    )

    # Propagate to other images
    other_images = [img for i, img in enumerate(images) if i != reference_idx]

    if other_images:
        other_masks = await loop.run_in_executor(
            None,
            propagate_mask_to_other_images,
            reference_image,
            reference_mask,
            other_images,
        )

        # Reconstruct full mask list in correct order
        masks = []
        other_idx = 0
        for i in range(len(images)):
            if i == reference_idx:
                masks.append(reference_mask)
            else:
                masks.append(other_masks[other_idx])
                other_idx += 1
    else:
        masks = [reference_mask]

    return masks
