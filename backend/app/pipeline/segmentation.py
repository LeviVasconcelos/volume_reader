import asyncio
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch

from app.models import ClickPoint, SegmentationPrompt, BoundingBox, BoardLandmarks


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


def segment_with_prompts(
    image: np.ndarray,
    foreground_points: list[tuple[float, float]] = None,
    background_points: list[tuple[float, float]] = None,
    bounding_box: Optional[BoundingBox] = None,
) -> np.ndarray:
    """
    Segment using SAM with multiple prompt types.

    Args:
        image: BGR image
        foreground_points: List of (x, y) normalized coords for foreground (board)
        background_points: List of (x, y) normalized coords for background (not board)
        bounding_box: Optional bounding box around the object

    Returns:
        Binary mask where object is 255, background is 0
    """
    predictor = load_sam_model()

    # Convert BGR to RGB for SAM
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    h, w = image.shape[:2]

    # Build point prompts
    input_points = []
    input_labels = []

    # Add foreground points (label = 1)
    if foreground_points:
        for (px, py) in foreground_points:
            input_points.append([int(px * w), int(py * h)])
            input_labels.append(1)

    # Add background points (label = 0)
    if background_points:
        for (px, py) in background_points:
            input_points.append([int(px * w), int(py * h)])
            input_labels.append(0)

    # Convert to numpy arrays
    input_point_array = np.array(input_points) if input_points else None
    input_label_array = np.array(input_labels) if input_labels else None

    # Build box prompt
    input_box = None
    if bounding_box:
        x1, y1, x2, y2 = bounding_box.to_pixels(w, h)
        input_box = np.array([x1, y1, x2, y2])

    # SAM requires at least one prompt
    if input_point_array is None and input_box is None:
        raise ValueError("At least one point or bounding box prompt is required")

    # Predict mask
    masks, scores, _ = predictor.predict(
        point_coords=input_point_array,
        point_labels=input_label_array,
        box=input_box,
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


def segment_single_image(
    image: np.ndarray,
    click_point: tuple[float, float],
) -> np.ndarray:
    """
    Segment the surfboard in a single image using SAM (legacy single-point version).

    Args:
        image: BGR image
        click_point: Normalized (x, y) coordinates where user clicked

    Returns:
        Binary mask where surfboard is 255, background is 0
    """
    return segment_with_prompts(
        image,
        foreground_points=[click_point],
    )


def segment_with_full_prompt(
    image: np.ndarray,
    prompt: SegmentationPrompt,
) -> np.ndarray:
    """
    Segment using a full SegmentationPrompt with all available prompts.

    Args:
        image: BGR image
        prompt: SegmentationPrompt with points, landmarks, box, etc.

    Returns:
        Binary mask
    """
    foreground_points = prompt.get_all_foreground_points()
    background_points = prompt.background_points if prompt.background_points else None

    return segment_with_prompts(
        image,
        foreground_points=foreground_points if foreground_points else None,
        background_points=background_points,
        bounding_box=prompt.bounding_box,
    )


def estimate_board_bbox_from_mask(mask: np.ndarray, padding: float = 0.05) -> BoundingBox:
    """
    Estimate bounding box from a segmentation mask.

    Args:
        mask: Binary mask
        padding: Padding to add as fraction of dimensions

    Returns:
        BoundingBox with normalized coordinates
    """
    h, w = mask.shape[:2]

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Return center box as fallback
        return BoundingBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)

    # Get bounding rect of largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    # Add padding
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x_min = max(0, x - pad_x) / w
    y_min = max(0, y - pad_y) / h
    x_max = min(w, x + bw + pad_x) / w
    y_max = min(h, y + bh + pad_y) / h

    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def estimate_landmarks_from_mask(mask: np.ndarray) -> BoardLandmarks:
    """
    Estimate board landmark points from a segmentation mask.

    Assumes surfboard is roughly vertical in image.

    Args:
        mask: Binary mask

    Returns:
        BoardLandmarks with estimated positions
    """
    h, w = mask.shape[:2]

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return BoardLandmarks(center=(0.5, 0.5))

    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    points = largest.reshape(-1, 2)

    # Find extremal points
    top_idx = np.argmin(points[:, 1])
    bottom_idx = np.argmax(points[:, 1])
    left_idx = np.argmin(points[:, 0])
    right_idx = np.argmax(points[:, 0])

    # Centroid
    M = cv2.moments(largest)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = w / 2, h / 2

    # Determine if board is more vertical or horizontal
    bbox = cv2.boundingRect(largest)
    is_vertical = bbox[3] > bbox[2]  # height > width

    if is_vertical:
        # Nose at top, tail at bottom
        nose = points[top_idx]
        tail = points[bottom_idx]
        # Rails on sides at center height
        mid_y = cy
        left_rail_pts = points[np.abs(points[:, 1] - mid_y) < h * 0.1]
        if len(left_rail_pts) > 0:
            rail_left = left_rail_pts[np.argmin(left_rail_pts[:, 0])]
            rail_right = left_rail_pts[np.argmax(left_rail_pts[:, 0])]
        else:
            rail_left = points[left_idx]
            rail_right = points[right_idx]
    else:
        # Nose on left, tail on right (or vice versa)
        nose = points[left_idx]
        tail = points[right_idx]
        # Rails on top/bottom
        mid_x = cx
        top_rail_pts = points[np.abs(points[:, 0] - mid_x) < w * 0.1]
        if len(top_rail_pts) > 0:
            rail_left = top_rail_pts[np.argmin(top_rail_pts[:, 1])]
            rail_right = top_rail_pts[np.argmax(top_rail_pts[:, 1])]
        else:
            rail_left = points[top_idx]
            rail_right = points[bottom_idx]

    return BoardLandmarks(
        nose=(float(nose[0] / w), float(nose[1] / h)),
        tail=(float(tail[0] / w), float(tail[1] / h)),
        rail_left=(float(rail_left[0] / w), float(rail_left[1] / h)),
        rail_right=(float(rail_right[0] / w), float(rail_right[1] / h)),
        center=(float(cx / w), float(cy / h)),
    )


def propagate_mask_to_other_images(
    reference_image: np.ndarray,
    reference_mask: np.ndarray,
    target_images: list[np.ndarray],
    use_landmarks: bool = True,
) -> list[np.ndarray]:
    """
    Propagate segmentation from reference image to other images.

    Uses landmarks estimated from reference mask to guide segmentation
    in other views.

    Args:
        reference_image: Reference image (BGR)
        reference_mask: Reference segmentation mask
        target_images: List of target images
        use_landmarks: Whether to use estimated landmarks

    Returns:
        List of masks for target images
    """
    masks = []

    # Estimate landmarks and bbox from reference
    if use_landmarks:
        landmarks = estimate_landmarks_from_mask(reference_mask)
        bbox = estimate_board_bbox_from_mask(reference_mask, padding=0.1)
    else:
        landmarks = None
        bbox = None

    # Get centroid as fallback
    moments = cv2.moments(reference_mask)
    if moments["m00"] > 0:
        ref_cx = moments["m10"] / moments["m00"]
        ref_cy = moments["m01"] / moments["m00"]
    else:
        h, w = reference_mask.shape[:2]
        ref_cx, ref_cy = w / 2, h / 2

    h, w = reference_image.shape[:2]
    center_norm = (ref_cx / w, ref_cy / h)

    for target_image in target_images:
        if use_landmarks and landmarks:
            # Use full prompt with landmarks and bbox
            prompt = SegmentationPrompt(
                landmarks=landmarks,
                bounding_box=bbox,
            )
            mask = segment_with_full_prompt(target_image, prompt)
        else:
            # Fallback to single point
            mask = segment_with_prompts(
                target_image,
                foreground_points=[center_norm],
            )
        masks.append(mask)

    return masks


async def segment_board(
    images: list[np.ndarray],
    prompt: Union[ClickPoint, SegmentationPrompt],
) -> list[np.ndarray]:
    """
    Segment the surfboard in all images.

    Supports both legacy ClickPoint and new SegmentationPrompt.

    Args:
        images: List of BGR images
        prompt: ClickPoint or SegmentationPrompt specifying how to segment

    Returns:
        List of binary masks for each image
    """
    loop = asyncio.get_event_loop()

    # Handle legacy ClickPoint
    if isinstance(prompt, ClickPoint):
        reference_idx = prompt.image_index
        reference_image = images[reference_idx]

        reference_mask = await loop.run_in_executor(
            None,
            segment_single_image,
            reference_image,
            (prompt.x, prompt.y),
        )
    else:
        # Full SegmentationPrompt
        reference_idx = prompt.image_index
        reference_image = images[reference_idx]

        reference_mask = await loop.run_in_executor(
            None,
            segment_with_full_prompt,
            reference_image,
            prompt,
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
            True,  # use_landmarks
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


async def segment_board_with_refinement(
    images: list[np.ndarray],
    initial_prompt: Union[ClickPoint, SegmentationPrompt],
    refinement_iterations: int = 1,
) -> list[np.ndarray]:
    """
    Segment with iterative refinement.

    First pass uses initial prompt, subsequent passes use estimated
    landmarks from previous masks for better coverage.

    Args:
        images: List of BGR images
        initial_prompt: Initial segmentation prompt
        refinement_iterations: Number of refinement passes

    Returns:
        List of refined masks
    """
    # Initial segmentation
    masks = await segment_board(images, initial_prompt)

    for _ in range(refinement_iterations):
        refined_masks = []

        for i, (image, mask) in enumerate(zip(images, masks)):
            # Estimate landmarks from current mask
            landmarks = estimate_landmarks_from_mask(mask)
            bbox = estimate_board_bbox_from_mask(mask, padding=0.08)

            # Re-segment with estimated prompts
            prompt = SegmentationPrompt(
                image_index=i,
                landmarks=landmarks,
                bounding_box=bbox,
            )

            new_mask = segment_with_full_prompt(image, prompt)

            # Keep mask with larger area (assuming we're trying to capture full board)
            if np.sum(new_mask > 127) > np.sum(mask > 127):
                refined_masks.append(new_mask)
            else:
                refined_masks.append(mask)

        masks = refined_masks

    return masks
