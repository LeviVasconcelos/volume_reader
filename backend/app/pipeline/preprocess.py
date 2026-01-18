import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 camera matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


def extract_exif_intrinsics(image_path: Path, width: int, height: int) -> Optional[CameraIntrinsics]:
    """
    Attempt to extract camera intrinsics from EXIF data.
    Returns None if not enough information available.
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if not exif_data:
                return None

            exif = {TAGS.get(k, k): v for k, v in exif_data.items()}

            focal_length = exif.get('FocalLength')
            focal_length_35mm = exif.get('FocalLengthIn35mmFilm')

            if focal_length_35mm:
                # Convert 35mm equivalent to pixels
                # 35mm sensor is 36mm wide
                sensor_width_mm = 36.0
                focal_length_mm = focal_length_35mm
                fx = fy = (focal_length_mm / sensor_width_mm) * width
            elif focal_length:
                # Use focal length with assumed sensor size
                # This is less accurate without sensor size
                focal_length_mm = float(focal_length)
                # Assume typical smartphone sensor ~5mm
                sensor_width_mm = 5.0
                fx = fy = (focal_length_mm / sensor_width_mm) * width
            else:
                return None

            return CameraIntrinsics(
                fx=fx,
                fy=fy,
                cx=width / 2,
                cy=height / 2,
                width=width,
                height=height,
            )
    except Exception:
        return None


def estimate_intrinsics(width: int, height: int) -> CameraIntrinsics:
    """
    Estimate camera intrinsics when EXIF is not available.
    Uses reasonable defaults for smartphone cameras.
    """
    # Assume ~60 degree horizontal FOV (typical smartphone)
    # fx = width / (2 * tan(fov/2))
    fov_rad = np.radians(60)
    fx = fy = width / (2 * np.tan(fov_rad / 2))

    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height,
    )


def load_and_preprocess_image(image_path: Path, max_dimension: int = 2048) -> np.ndarray:
    """
    Load image and resize if needed to limit processing time.
    Returns BGR image (OpenCV format).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    h, w = img.shape[:2]

    # Resize if too large
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


async def preprocess_images(
    image_paths: list[Path],
    max_dimension: int = 2048,
) -> tuple[list[np.ndarray], CameraIntrinsics]:
    """
    Load and preprocess all images.

    Returns:
        images: List of preprocessed images (BGR format)
        camera_intrinsics: Estimated or extracted camera intrinsics
    """
    loop = asyncio.get_event_loop()

    # Load images in parallel using thread pool
    images = await asyncio.gather(*[
        loop.run_in_executor(None, load_and_preprocess_image, path, max_dimension)
        for path in image_paths
    ])

    # Verify all images have same dimensions
    shapes = [img.shape[:2] for img in images]
    if len(set(shapes)) > 1:
        # Resize all to match the first image
        target_h, target_w = shapes[0]
        images = [
            cv2.resize(img, (target_w, target_h)) if img.shape[:2] != (target_h, target_w) else img
            for img in images
        ]

    h, w = images[0].shape[:2]

    # Try to get intrinsics from EXIF of first image
    intrinsics = await loop.run_in_executor(
        None, extract_exif_intrinsics, image_paths[0], w, h
    )

    if intrinsics is None:
        intrinsics = estimate_intrinsics(w, h)

    return list(images), intrinsics
