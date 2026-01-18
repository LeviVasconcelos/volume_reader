import asyncio
from typing import Optional

import cv2
import numpy as np

from app.models import ScaleMethod, ArucoScaleData, CreditCardScaleData, UserDimensionsScaleData


# Standard credit card dimensions in mm (ISO/IEC 7810 ID-1)
CREDIT_CARD_WIDTH_MM = 85.6
CREDIT_CARD_HEIGHT_MM = 53.98
CREDIT_CARD_ASPECT_RATIO = CREDIT_CARD_WIDTH_MM / CREDIT_CARD_HEIGHT_MM


def detect_aruco_scale(
    images: list[np.ndarray],
    aruco_data: ArucoScaleData,
) -> tuple[Optional[float], list[str]]:
    """
    Detect ArUco marker and compute scale factor.

    Returns:
        scale_factor: Pixels to mm conversion factor, or None if not detected
        warnings: List of warning messages
    """
    warnings = []

    # Initialize ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    detected_scales = []

    for i, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # Find the specific marker we're looking for
            for j, marker_id in enumerate(ids):
                if marker_id[0] == aruco_data.aruco_id:
                    # Calculate marker size in pixels
                    marker_corners = corners[j][0]
                    # Average of all edge lengths
                    edge_lengths = [
                        np.linalg.norm(marker_corners[k] - marker_corners[(k + 1) % 4])
                        for k in range(4)
                    ]
                    marker_size_px = np.mean(edge_lengths)

                    # Scale factor: mm per pixel
                    scale = aruco_data.aruco_size_mm / marker_size_px
                    detected_scales.append(scale)
                    break

    if not detected_scales:
        warnings.append("ArUco marker not detected in any image")
        return None, warnings

    if len(detected_scales) < len(images) / 2:
        warnings.append(
            f"ArUco marker only detected in {len(detected_scales)}/{len(images)} images"
        )

    # Use median scale to be robust to outliers
    scale_factor = float(np.median(detected_scales))

    # Check for consistency
    scale_std = np.std(detected_scales)
    if scale_std / scale_factor > 0.1:  # More than 10% variation
        warnings.append("Inconsistent ArUco marker detection across images")

    return scale_factor, warnings


def detect_credit_card_scale(
    images: list[np.ndarray],
    masks: list[np.ndarray],
) -> tuple[Optional[float], list[str]]:
    """
    Detect credit card and compute scale factor.

    Looks for rectangular objects with credit card aspect ratio outside the board mask.

    Returns:
        scale_factor: Pixels to mm conversion factor, or None if not detected
        warnings: List of warning messages
    """
    warnings = []
    detected_scales = []

    for i, (image, mask) in enumerate(zip(images, masks)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Look outside the surfboard mask for the card
        search_mask = cv2.bitwise_not(mask)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=search_mask)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Credit card should be roughly rectangular (4 corners)
            if len(approx) != 4:
                continue

            # Get bounding rectangle
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]

            if width == 0 or height == 0:
                continue

            # Ensure width > height
            if width < height:
                width, height = height, width

            aspect_ratio = width / height

            # Check if aspect ratio matches credit card (within 10%)
            if abs(aspect_ratio - CREDIT_CARD_ASPECT_RATIO) < 0.15:
                # Check minimum size (card should be reasonably visible)
                area = width * height
                image_area = image.shape[0] * image.shape[1]
                if area < image_area * 0.001:  # Less than 0.1% of image
                    continue

                # Calculate scale: longer edge is 85.6mm
                scale = CREDIT_CARD_WIDTH_MM / width
                detected_scales.append(scale)

    if not detected_scales:
        warnings.append("Credit card not detected in any image")
        return None, warnings

    if len(detected_scales) < len(images) / 3:
        warnings.append(
            f"Credit card only detected in {len(detected_scales)}/{len(images)} images"
        )

    # Use median scale
    scale_factor = float(np.median(detected_scales))

    return scale_factor, warnings


def compute_scale_from_dimensions(
    user_dimensions: UserDimensionsScaleData,
) -> tuple[float, list[str]]:
    """
    Compute a placeholder scale factor when using user dimensions.

    The actual scaling will be applied after reconstruction by fitting
    the model to the provided dimensions.

    Returns:
        scale_factor: Set to 1.0 as a placeholder
        warnings: List of warning messages
    """
    warnings = []

    # Count how many dimensions were provided
    provided = sum([
        user_dimensions.length_mm is not None,
        user_dimensions.width_mm is not None,
        user_dimensions.thickness_mm is not None,
    ])

    if provided == 0:
        warnings.append("No dimensions provided for scaling")
        return 1.0, warnings

    if provided < 2:
        warnings.append(
            "Only one dimension provided; accuracy may be reduced. "
            "Providing length and width is recommended."
        )

    # Return 1.0 as placeholder; actual scaling happens in volume calculation
    return 1.0, warnings


async def detect_scale(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    scale_method: ScaleMethod,
    aruco_data: Optional[ArucoScaleData],
    credit_card_data: Optional[CreditCardScaleData],
    user_dimensions: Optional[UserDimensionsScaleData],
) -> tuple[float, list[str]]:
    """
    Detect scale reference based on the specified method.

    Returns:
        scale_factor: Conversion factor from pixels to mm (or 1.0 for user dimensions)
        warnings: List of warning messages
    """
    loop = asyncio.get_event_loop()

    if scale_method == ScaleMethod.ARUCO:
        if aruco_data is None:
            return 1.0, ["ArUco method selected but no aruco_data provided"]
        return await loop.run_in_executor(
            None, detect_aruco_scale, images, aruco_data
        )

    elif scale_method == ScaleMethod.CREDIT_CARD:
        return await loop.run_in_executor(
            None, detect_credit_card_scale, images, masks
        )

    elif scale_method == ScaleMethod.USER_DIMENSIONS:
        if user_dimensions is None:
            return 1.0, ["User dimensions method selected but no dimensions provided"]
        return compute_scale_from_dimensions(user_dimensions)

    return 1.0, [f"Unknown scale method: {scale_method}"]
