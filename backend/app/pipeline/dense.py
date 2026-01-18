import asyncio
from typing import Optional

import cv2
import numpy as np
import open3d as o3d

from app.pipeline.sfm import CameraPose
from app.pipeline.preprocess import CameraIntrinsics


def compute_depth_map(
    image1: np.ndarray,
    image2: np.ndarray,
    mask1: np.ndarray,
    pose1: CameraPose,
    pose2: CameraPose,
    K: np.ndarray,
) -> np.ndarray:
    """
    Compute depth map using stereo matching.

    Args:
        image1, image2: Input images (BGR)
        mask1: Segmentation mask for image1
        pose1, pose2: Camera poses
        K: Camera intrinsic matrix

    Returns:
        depth_map: Depth map for image1
    """
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute relative pose
    R_rel = pose2.R @ pose1.R.T
    t_rel = pose2.t - R_rel @ pose1.t

    # Compute rectification transforms
    h, w = gray1.shape[:2]

    # Get rectification matrices
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, None, K, None,
        (w, h), R_rel, t_rel,
        alpha=0,
    )

    # Compute rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(K, None, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, None, R2, P2, (w, h), cv2.CV_32FC1)

    # Rectify images
    rect1 = cv2.remap(gray1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(gray2, map2x, map2y, cv2.INTER_LINEAR)

    # Stereo matching
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )

    disparity = stereo.compute(rect1, rect2).astype(np.float32) / 16.0

    # Convert disparity to depth
    # depth = baseline * focal_length / disparity
    baseline = np.linalg.norm(t_rel)
    focal_length = K[0, 0]

    with np.errstate(divide='ignore', invalid='ignore'):
        depth_map = (baseline * focal_length) / disparity
        depth_map[disparity <= 0] = 0
        depth_map[np.isinf(depth_map)] = 0

    # Apply mask
    depth_map = depth_map * (mask1 > 0).astype(np.float32)

    return depth_map


def depth_to_point_cloud(
    depth_map: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    pose: CameraPose,
) -> o3d.geometry.PointCloud:
    """
    Convert depth map to colored point cloud.

    Args:
        depth_map: Depth values per pixel
        image: Color image (BGR)
        mask: Segmentation mask
        K: Camera intrinsic matrix
        pose: Camera pose

    Returns:
        point_cloud: Open3D point cloud
    """
    h, w = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Valid depth mask
    valid = (depth_map > 0) & (depth_map < 100) & (mask > 0)

    # Get valid pixels
    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = depth_map[valid]

    # Back-project to 3D (camera coordinates)
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    z = z_valid

    points_cam = np.stack([x, y, z], axis=-1)

    # Transform to world coordinates
    points_world = (pose.R.T @ (points_cam.T - pose.t)).T

    # Get colors (BGR to RGB)
    colors = image[valid][:, ::-1] / 255.0

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def merge_point_clouds(
    point_clouds: list[o3d.geometry.PointCloud],
    voxel_size: float = 0.002,
) -> o3d.geometry.PointCloud:
    """
    Merge multiple point clouds and remove duplicates.

    Args:
        point_clouds: List of point clouds to merge
        voxel_size: Voxel size for downsampling (removes duplicates)

    Returns:
        merged: Merged and cleaned point cloud
    """
    if not point_clouds:
        return o3d.geometry.PointCloud()

    # Combine all point clouds
    merged = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        merged += pcd

    # Downsample to remove duplicates
    merged = merged.voxel_down_sample(voxel_size)

    # Remove outliers
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Estimate normals
    merged.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    merged.orient_normals_consistent_tangent_plane(k=15)

    return merged


async def densify_point_cloud(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    camera_poses: list[CameraPose],
    intrinsics: CameraIntrinsics,
) -> tuple[o3d.geometry.PointCloud, list[str]]:
    """
    Create dense point cloud from images using multi-view stereo.

    Args:
        images: List of BGR images
        masks: Segmentation masks
        camera_poses: Camera poses from SfM
        intrinsics: Camera intrinsics

    Returns:
        dense_cloud: Dense point cloud
        warnings: List of warnings
    """
    loop = asyncio.get_event_loop()
    warnings = []
    K = intrinsics.to_matrix()

    point_clouds = []

    # Process image pairs
    for i, pose1 in enumerate(camera_poses):
        if pose1.image_idx >= len(images):
            continue

        # Find neighboring camera
        for j, pose2 in enumerate(camera_poses):
            if i >= j or pose2.image_idx >= len(images):
                continue

            img_idx1 = pose1.image_idx
            img_idx2 = pose2.image_idx

            # Compute depth map
            try:
                depth_map = await loop.run_in_executor(
                    None,
                    compute_depth_map,
                    images[img_idx1],
                    images[img_idx2],
                    masks[img_idx1],
                    pose1,
                    pose2,
                    K,
                )

                # Convert to point cloud
                pcd = await loop.run_in_executor(
                    None,
                    depth_to_point_cloud,
                    depth_map,
                    images[img_idx1],
                    masks[img_idx1],
                    K,
                    pose1,
                )

                if len(pcd.points) > 100:
                    point_clouds.append(pcd)
            except Exception as e:
                warnings.append(f"Depth estimation failed for pair {i}-{j}: {str(e)}")

    if not point_clouds:
        warnings.append("Dense reconstruction produced no points - using sparse cloud")
        # Return empty cloud, will fallback to sparse
        return o3d.geometry.PointCloud(), warnings

    # Merge all point clouds
    dense_cloud = await loop.run_in_executor(
        None,
        merge_point_clouds,
        point_clouds,
    )

    if len(dense_cloud.points) < 1000:
        warnings.append("Dense point cloud has low density - results may be less accurate")

    return dense_cloud, warnings
