import numpy as np
import pytest
import cv2

from app.pipeline.features import (
    extract_sift_features,
    match_features,
    match_all_pairs,
    geometric_verification,
    extract_and_match_features,
)


class TestSiftExtraction:
    def test_extract_features_from_textured_image(self):
        # Create image with texture (checkerboard pattern)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(0, 480, 20):
            for j in range(0, 640, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    img[i:i+20, j:j+20] = 255

        keypoints, descriptors = extract_sift_features(img)

        assert len(keypoints) > 0
        assert descriptors.shape[0] == len(keypoints)
        assert descriptors.shape[1] == 128  # SIFT descriptor size

    def test_extract_features_with_mask(self):
        # Create textured image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Mask only allows features in top-left quadrant
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[:240, :320] = 255

        keypoints, descriptors = extract_sift_features(img, mask=mask)

        # All keypoints should be within masked region
        for kp in keypoints:
            assert kp.pt[0] < 320
            assert kp.pt[1] < 240

    def test_extract_features_blank_image(self):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128

        keypoints, descriptors = extract_sift_features(img)

        # Blank image should have few or no features
        assert len(keypoints) < 10


class TestFeatureMatching:
    def test_match_identical_descriptors(self):
        # Create random descriptors
        desc = np.random.rand(100, 128).astype(np.float32)

        # Matching with itself should give perfect matches
        matches = match_features(desc, desc, ratio_threshold=0.9)

        assert len(matches) > 50

    def test_match_different_descriptors(self):
        desc1 = np.random.rand(100, 128).astype(np.float32)
        desc2 = np.random.rand(100, 128).astype(np.float32)

        matches = match_features(desc1, desc2)

        # Random descriptors should have few good matches
        assert len(matches) < 30

    def test_match_empty_descriptors(self):
        desc1 = np.array([]).reshape(0, 128).astype(np.float32)
        desc2 = np.random.rand(100, 128).astype(np.float32)

        matches = match_features(desc1, desc2)

        assert len(matches) == 0


class TestMatchAllPairs:
    def test_match_three_images(self):
        # Create 3 sets of descriptors
        np.random.seed(42)
        base_desc = np.random.rand(100, 128).astype(np.float32)

        # Add some noise to create related descriptors
        desc1 = base_desc + np.random.randn(100, 128).astype(np.float32) * 0.1
        desc2 = base_desc + np.random.randn(100, 128).astype(np.float32) * 0.1
        desc3 = base_desc + np.random.randn(100, 128).astype(np.float32) * 0.1

        all_matches = match_all_pairs([desc1, desc2, desc3], min_matches=10)

        # Should have matches for pairs (0,1), (0,2), (1,2)
        assert isinstance(all_matches, dict)
        # Some pairs should have matches
        for key in all_matches:
            assert key[0] < key[1]  # Pairs are ordered


class TestGeometricVerification:
    def test_verification_with_inliers(self):
        # Create synthetic corresponding points with some outliers
        np.random.seed(42)

        # Generate points in image 1
        pts1 = np.random.rand(50, 2) * 400 + 100
        kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts1]

        # Transform points (simple translation + small noise for inliers)
        pts2 = pts1 + np.array([50, 30]) + np.random.randn(50, 2) * 2
        # Add outliers
        pts2[-5:] = np.random.rand(5, 2) * 600
        kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts2]

        # Create matches
        matches = [cv2.DMatch(i, i, 0) for i in range(50)]

        inliers, F = geometric_verification(kp1, kp2, matches)

        # Should have fewer inliers than total matches (outliers removed)
        assert len(inliers) < len(matches)
        assert len(inliers) >= 8  # At least enough for fundamental matrix

    def test_verification_few_matches(self):
        kp1 = [cv2.KeyPoint(float(i), float(i), 1) for i in range(5)]
        kp2 = [cv2.KeyPoint(float(i+10), float(i+10), 1) for i in range(5)]
        matches = [cv2.DMatch(i, i, 0) for i in range(5)]

        inliers, F = geometric_verification(kp1, kp2, matches)

        # Too few matches, should return original
        assert len(inliers) == len(matches)
        assert F is None


@pytest.mark.asyncio
async def test_extract_and_match_features():
    # Create two related images (shifted checkerboard)
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(0, 480, 30):
        for j in range(0, 640, 30):
            if (i // 30 + j // 30) % 2 == 0:
                img1[i:i+30, j:j+30] = 255
                # Shifted version
                if i+5 < 480 and j+5 < 640:
                    img2[i+5:min(i+35, 480), j+5:min(j+35, 640)] = 255

    masks = [
        np.ones((480, 640), dtype=np.uint8) * 255,
        np.ones((480, 640), dtype=np.uint8) * 255,
    ]

    keypoints, descriptors, matches = await extract_and_match_features(
        [img1, img2], masks
    )

    assert len(keypoints) == 2
    assert len(descriptors) == 2
    assert isinstance(matches, dict)
