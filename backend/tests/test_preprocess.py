import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.pipeline.preprocess import (
    CameraIntrinsics,
    estimate_intrinsics,
    load_and_preprocess_image,
    preprocess_images,
)


class TestCameraIntrinsics:
    def test_to_matrix(self):
        intrinsics = CameraIntrinsics(
            fx=1000, fy=1000, cx=320, cy=240, width=640, height=480
        )
        K = intrinsics.to_matrix()

        assert K.shape == (3, 3)
        assert K[0, 0] == 1000  # fx
        assert K[1, 1] == 1000  # fy
        assert K[0, 2] == 320   # cx
        assert K[1, 2] == 240   # cy
        assert K[2, 2] == 1


class TestEstimateIntrinsics:
    def test_estimate_intrinsics_standard_resolution(self):
        intrinsics = estimate_intrinsics(1920, 1080)

        assert intrinsics.width == 1920
        assert intrinsics.height == 1080
        assert intrinsics.cx == 960  # width / 2
        assert intrinsics.cy == 540  # height / 2
        # Focal length should be reasonable for 60 degree FOV
        assert 1500 < intrinsics.fx < 2000

    def test_estimate_intrinsics_square(self):
        intrinsics = estimate_intrinsics(1000, 1000)

        assert intrinsics.cx == 500
        assert intrinsics.cy == 500
        assert intrinsics.fx == intrinsics.fy


class TestLoadAndPreprocessImage:
    def test_load_nonexistent_image(self, tmp_path):
        fake_path = tmp_path / "nonexistent.jpg"

        with pytest.raises(ValueError, match="Failed to load image"):
            load_and_preprocess_image(fake_path)

    def test_load_and_resize_large_image(self, tmp_path):
        import cv2

        # Create a large test image
        large_img = np.random.randint(0, 255, (4000, 3000, 3), dtype=np.uint8)
        img_path = tmp_path / "large.jpg"
        cv2.imwrite(str(img_path), large_img)

        # Load with max dimension 2048
        result = load_and_preprocess_image(img_path, max_dimension=2048)

        assert max(result.shape[:2]) <= 2048
        assert result.shape[2] == 3  # Still BGR

    def test_load_small_image_unchanged(self, tmp_path):
        import cv2

        # Create a small test image
        small_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = tmp_path / "small.jpg"
        cv2.imwrite(str(img_path), small_img)

        result = load_and_preprocess_image(img_path, max_dimension=2048)

        assert result.shape == (480, 640, 3)


@pytest.mark.asyncio
async def test_preprocess_images(tmp_path):
    import cv2

    # Create test images
    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    path1 = tmp_path / "img1.jpg"
    path2 = tmp_path / "img2.jpg"
    cv2.imwrite(str(path1), img1)
    cv2.imwrite(str(path2), img2)

    images, intrinsics = await preprocess_images([path1, path2])

    assert len(images) == 2
    assert all(img.shape == (480, 640, 3) for img in images)
    assert intrinsics.width == 640
    assert intrinsics.height == 480
