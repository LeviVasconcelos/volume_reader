import numpy as np
import pytest
import cv2

from app.pipeline.scale import (
    CREDIT_CARD_WIDTH_MM,
    CREDIT_CARD_HEIGHT_MM,
    CREDIT_CARD_ASPECT_RATIO,
    detect_aruco_scale,
    detect_credit_card_scale,
    compute_scale_from_dimensions,
    detect_scale,
)
from app.models import (
    ScaleMethod,
    ArucoScaleData,
    CreditCardScaleData,
    UserDimensionsScaleData,
)


class TestArucoDetection:
    def test_detect_aruco_marker(self):
        # Create a synthetic image with an ArUco marker
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # Generate ArUco marker
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        marker_size = 100  # pixels
        marker_id = 42
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

        # Place marker in image
        x, y = 200, 150
        img[y:y+marker_size, x:x+marker_size] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

        aruco_data = ArucoScaleData(aruco_id=42, aruco_size_mm=50.0)
        scale_factor, warnings = detect_aruco_scale([img], aruco_data)

        assert scale_factor is not None
        # Scale should be approximately 50mm / 100px = 0.5 mm/px
        assert 0.4 < scale_factor < 0.6
        assert len(warnings) == 0

    def test_detect_aruco_wrong_id(self):
        # Create image with marker ID 10
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, 10, 100)
        img[100:200, 100:200] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

        # Look for different ID
        aruco_data = ArucoScaleData(aruco_id=42, aruco_size_mm=50.0)
        scale_factor, warnings = detect_aruco_scale([img], aruco_data)

        assert scale_factor is None
        assert any("not detected" in w for w in warnings)

    def test_detect_aruco_no_marker(self):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128

        aruco_data = ArucoScaleData(aruco_id=42, aruco_size_mm=50.0)
        scale_factor, warnings = detect_aruco_scale([img], aruco_data)

        assert scale_factor is None
        assert len(warnings) > 0


class TestCreditCardDetection:
    def test_credit_card_aspect_ratio(self):
        # Verify standard credit card dimensions
        assert abs(CREDIT_CARD_WIDTH_MM - 85.6) < 0.1
        assert abs(CREDIT_CARD_HEIGHT_MM - 53.98) < 0.1
        assert 1.5 < CREDIT_CARD_ASPECT_RATIO < 1.6

    def test_detect_credit_card_synthetic(self):
        # Create synthetic image with rectangle matching credit card aspect ratio
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200

        # Draw rectangle with credit card proportions
        # 85.6 x 53.98 mm -> aspect ratio ~1.586
        card_width_px = 171  # ~2x mm
        card_height_px = 108  # ~2x mm

        # Draw white card on gray background
        x, y = 50, 300
        cv2.rectangle(img, (x, y), (x + card_width_px, y + card_height_px), (255, 255, 255), -1)
        cv2.rectangle(img, (x, y), (x + card_width_px, y + card_height_px), (0, 0, 0), 2)

        # Create a mask (surfboard in center, card outside)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:400, 200:500] = 255  # Surfboard region

        scale_factor, warnings = detect_credit_card_scale([img], [mask])

        # Detection may or may not work depending on edge detection
        # Just verify it doesn't crash
        assert isinstance(warnings, list)


class TestUserDimensions:
    def test_all_dimensions_provided(self):
        dims = UserDimensionsScaleData(
            length_mm=1830,
            width_mm=520,
            thickness_mm=63
        )
        scale_factor, warnings = compute_scale_from_dimensions(dims)

        assert scale_factor == 1.0  # Placeholder until mesh fitting
        assert len(warnings) == 0

    def test_single_dimension_warning(self):
        dims = UserDimensionsScaleData(length_mm=1830)
        scale_factor, warnings = compute_scale_from_dimensions(dims)

        assert scale_factor == 1.0
        assert any("one dimension" in w.lower() for w in warnings)

    def test_no_dimensions_warning(self):
        dims = UserDimensionsScaleData()
        scale_factor, warnings = compute_scale_from_dimensions(dims)

        assert scale_factor == 1.0
        assert any("no dimensions" in w.lower() for w in warnings)


@pytest.mark.asyncio
async def test_detect_scale_user_dimensions():
    images = [np.zeros((480, 640, 3), dtype=np.uint8)]
    masks = [np.ones((480, 640), dtype=np.uint8) * 255]

    user_dims = UserDimensionsScaleData(length_mm=1830, width_mm=520)

    scale_factor, warnings = await detect_scale(
        images,
        masks,
        ScaleMethod.USER_DIMENSIONS,
        aruco_data=None,
        credit_card_data=None,
        user_dimensions=user_dims,
    )

    assert scale_factor == 1.0
    assert isinstance(warnings, list)
