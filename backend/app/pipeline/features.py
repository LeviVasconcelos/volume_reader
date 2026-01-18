import asyncio
from typing import Optional

import cv2
import numpy as np


def extract_sift_features(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_features: int = 8000,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """
    Extract SIFT features from an image.

    Args:
        image: BGR image
        mask: Optional mask to restrict feature detection
        max_features: Maximum number of features to extract

    Returns:
        keypoints: List of detected keypoints
        descriptors: Feature descriptors (N x 128)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=max_features)

    # Detect and compute
    keypoints, descriptors = sift.detectAndCompute(gray, mask)

    if descriptors is None:
        return [], np.array([])

    return keypoints, descriptors


def match_features(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_threshold: float = 0.75,
) -> list[cv2.DMatch]:
    """
    Match features between two images using FLANN matcher with ratio test.

    Args:
        desc1: Descriptors from first image
        desc2: Descriptors from second image
        ratio_threshold: Lowe's ratio test threshold

    Returns:
        good_matches: List of good matches passing ratio test
    """
    if len(desc1) < 2 or len(desc2) < 2:
        return []

    # FLANN parameters for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find 2 nearest neighbors for ratio test
    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return []

    # Apply ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return good_matches


def match_all_pairs(
    descriptors: list[np.ndarray],
    min_matches: int = 30,
) -> dict[tuple[int, int], list[cv2.DMatch]]:
    """
    Match features between all image pairs.

    Args:
        descriptors: List of descriptors for each image
        min_matches: Minimum matches required to keep a pair

    Returns:
        matches: Dictionary mapping (i, j) pairs to their matches
    """
    n_images = len(descriptors)
    all_matches = {}

    for i in range(n_images):
        for j in range(i + 1, n_images):
            matches = match_features(descriptors[i], descriptors[j])

            if len(matches) >= min_matches:
                all_matches[(i, j)] = matches

    return all_matches


def geometric_verification(
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    ransac_threshold: float = 4.0,
) -> tuple[list[cv2.DMatch], Optional[np.ndarray]]:
    """
    Verify matches geometrically using fundamental matrix estimation.

    Args:
        kp1: Keypoints from first image
        kp2: Keypoints from second image
        matches: Initial matches
        ransac_threshold: RANSAC inlier threshold in pixels

    Returns:
        inlier_matches: Matches that are geometric inliers
        F: Fundamental matrix (or None if estimation failed)
    """
    if len(matches) < 8:
        return matches, None

    # Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Estimate fundamental matrix
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        cv2.FM_RANSAC,
        ransac_threshold,
        0.99
    )

    if F is None or mask is None:
        return matches, None

    # Filter to inliers only
    inlier_matches = [m for m, is_inlier in zip(matches, mask.ravel()) if is_inlier]

    return inlier_matches, F


async def extract_and_match_features(
    images: list[np.ndarray],
    masks: list[np.ndarray],
) -> tuple[list[list[cv2.KeyPoint]], list[np.ndarray], dict[tuple[int, int], list[cv2.DMatch]]]:
    """
    Extract features from all images and match them pairwise.

    Args:
        images: List of BGR images
        masks: List of segmentation masks

    Returns:
        keypoints: List of keypoints for each image
        descriptors: List of descriptors for each image
        matches: Dictionary of verified matches between image pairs
    """
    loop = asyncio.get_event_loop()

    # Extract features from all images in parallel
    feature_tasks = [
        loop.run_in_executor(None, extract_sift_features, img, mask)
        for img, mask in zip(images, masks)
    ]
    features = await asyncio.gather(*feature_tasks)

    keypoints = [f[0] for f in features]
    descriptors = [f[1] for f in features]

    # Match all pairs
    raw_matches = await loop.run_in_executor(
        None, match_all_pairs, descriptors
    )

    # Geometric verification for each pair
    verified_matches = {}
    for (i, j), matches in raw_matches.items():
        inliers, _ = await loop.run_in_executor(
            None,
            geometric_verification,
            keypoints[i],
            keypoints[j],
            matches,
        )
        if len(inliers) >= 20:  # Keep pairs with enough inliers
            verified_matches[(i, j)] = inliers

    return keypoints, descriptors, verified_matches
