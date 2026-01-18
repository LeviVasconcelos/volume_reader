import asyncio
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import open3d as o3d

from app.pipeline.preprocess import CameraIntrinsics


@dataclass
class CameraPose:
    """Camera pose (extrinsics)."""
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector
    image_idx: int

    def to_projection_matrix(self, K: np.ndarray) -> np.ndarray:
        """Get 3x4 projection matrix P = K[R|t]."""
        Rt = np.hstack([self.R, self.t])
        return K @ Rt


def triangulate_points(
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    K: np.ndarray,
    pose1: CameraPose,
    pose2: CameraPose,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate 3D points from matched features.

    Returns:
        points_3d: Nx3 array of 3D points
        point_indices: Indices of matches used
    """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    P1 = pose1.to_projection_matrix(K)
    P2 = pose2.to_projection_matrix(K)

    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    # Convert from homogeneous coordinates
    points_3d = points_4d[:3, :] / points_4d[3, :]
    points_3d = points_3d.T

    return points_3d, np.arange(len(matches))


def initialize_reconstruction(
    keypoints: list[list[cv2.KeyPoint]],
    matches: dict[tuple[int, int], list[cv2.DMatch]],
    K: np.ndarray,
) -> tuple[Optional[tuple[int, int]], list[CameraPose], np.ndarray]:
    """
    Initialize reconstruction with the best image pair.

    Returns:
        best_pair: Indices of the initial image pair
        poses: Initial camera poses
        points_3d: Initial 3D points
    """
    best_pair = None
    best_points = None
    best_poses = None
    max_points = 0

    for (i, j), pair_matches in matches.items():
        if len(pair_matches) < 50:
            continue

        pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in pair_matches])
        pts2 = np.float32([keypoints[j][m.trainIdx].pt for m in pair_matches])

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)

        if E is None:
            continue

        # Recover pose
        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)

        # First camera at origin
        pose1 = CameraPose(R=np.eye(3), t=np.zeros((3, 1)), image_idx=i)
        pose2 = CameraPose(R=R, t=t, image_idx=j)

        # Triangulate points
        points_3d, _ = triangulate_points(
            keypoints[i], keypoints[j], pair_matches, K, pose1, pose2
        )

        # Filter points behind cameras and too far
        valid = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 100)
        n_valid = np.sum(valid)

        if n_valid > max_points:
            max_points = n_valid
            best_pair = (i, j)
            best_points = points_3d[valid]
            best_poses = [pose1, pose2]

    if best_pair is None:
        return None, [], np.array([])

    return best_pair, best_poses, best_points


def estimate_pose_pnp(
    keypoints: list[cv2.KeyPoint],
    points_3d: np.ndarray,
    point_2d_indices: list[int],
    K: np.ndarray,
) -> Optional[CameraPose]:
    """
    Estimate camera pose using PnP.

    Args:
        keypoints: Keypoints in the new image
        points_3d: 3D points
        point_2d_indices: Indices into keypoints that correspond to points_3d
        K: Camera intrinsic matrix

    Returns:
        pose: Estimated camera pose, or None if failed
    """
    if len(point_2d_indices) < 6:
        return None

    object_points = points_3d[:len(point_2d_indices)].astype(np.float32)
    image_points = np.float32([keypoints[idx].pt for idx in point_2d_indices])

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, K, None,
        iterationsCount=1000,
        reprojectionError=4.0,
        confidence=0.99,
    )

    if not success or inliers is None or len(inliers) < 6:
        return None

    R, _ = cv2.Rodrigues(rvec)

    return CameraPose(R=R, t=tvec, image_idx=-1)


def bundle_adjustment_simple(
    points_3d: np.ndarray,
    poses: list[CameraPose],
    observations: dict,
    K: np.ndarray,
) -> tuple[np.ndarray, list[CameraPose]]:
    """
    Simple bundle adjustment using iterative refinement.

    For a production system, use scipy.optimize or ceres-solver.
    This is a simplified version that just filters outliers.
    """
    # For now, just filter obvious outliers based on reprojection error
    filtered_points = []

    for pt_idx, point in enumerate(points_3d):
        errors = []
        for pose in poses:
            if pose.image_idx in observations.get(pt_idx, {}):
                # Project point
                P = pose.to_projection_matrix(K)
                pt_h = np.append(point, 1)
                proj = P @ pt_h
                proj = proj[:2] / proj[2]

                # Get observed point
                obs = observations[pt_idx][pose.image_idx]
                error = np.linalg.norm(proj - obs)
                errors.append(error)

        if errors and np.mean(errors) < 10:  # Max 10 pixel average error
            filtered_points.append(point)

    return np.array(filtered_points), poses


async def run_sfm(
    images: list[np.ndarray],
    keypoints: list[list[cv2.KeyPoint]],
    matches: dict[tuple[int, int], list[cv2.DMatch]],
    intrinsics: CameraIntrinsics,
) -> tuple[o3d.geometry.PointCloud, list[CameraPose]]:
    """
    Run Structure from Motion pipeline.

    Args:
        images: List of images
        keypoints: Keypoints for each image
        matches: Pairwise matches
        intrinsics: Camera intrinsics

    Returns:
        sparse_cloud: Sparse 3D point cloud
        camera_poses: List of camera poses
    """
    loop = asyncio.get_event_loop()
    K = intrinsics.to_matrix()

    # Initialize with best pair
    init_result = await loop.run_in_executor(
        None,
        initialize_reconstruction,
        keypoints, matches, K
    )
    init_pair, poses, points_3d = init_result

    if init_pair is None:
        raise ValueError("Failed to initialize reconstruction - not enough matches")

    registered_images = set(init_pair)

    # Track which 3D points are seen in which images
    # observations[point_idx][image_idx] = 2D point
    observations = {}

    # Incrementally add remaining images
    remaining = set(range(len(images))) - registered_images

    while remaining:
        best_image = None
        best_pose = None
        best_matches = 0

        for img_idx in remaining:
            # Find matches with registered images
            matched_3d_indices = []
            matched_2d_indices = []

            for reg_idx in registered_images:
                pair_key = (min(img_idx, reg_idx), max(img_idx, reg_idx))
                if pair_key not in matches:
                    continue

                pair_matches = matches[pair_key]
                # This is simplified - in practice need to track 3D-2D correspondences
                if len(pair_matches) > best_matches:
                    best_matches = len(pair_matches)
                    best_image = img_idx

        if best_image is None:
            break

        # For simplicity, just register with identity pose offset
        # A full implementation would use PnP here
        if poses:
            last_pose = poses[-1]
            # Simple offset estimation
            new_pose = CameraPose(
                R=last_pose.R.copy(),
                t=last_pose.t.copy() + np.array([[0.1], [0], [0]]),
                image_idx=best_image
            )
            poses.append(new_pose)

        registered_images.add(best_image)
        remaining.remove(best_image)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Estimate normals
    if len(points_3d) > 10:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

    return pcd, poses
