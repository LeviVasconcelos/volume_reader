"""Debug visualizations for object verification (match-based vs SAM segmentation)."""

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from app.pipeline.debug import DebugContext
    from app.pipeline.sfm import CameraPose
    from app.pipeline.preprocess import CameraIntrinsics


def project_points_to_image(
    points_3d: np.ndarray,
    camera_pose: "CameraPose",
    intrinsics: "CameraIntrinsics",
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Project 3D points to 2D image coordinates.

    Args:
        points_3d: Nx3 array of 3D points
        camera_pose: Camera pose (R, t)
        intrinsics: Camera intrinsics
        image_shape: (height, width)

    Returns:
        Nx2 array of valid 2D points (filtered to image bounds)
    """
    if len(points_3d) == 0:
        return np.array([]).reshape(0, 2)

    K = intrinsics.to_matrix()
    R = camera_pose.R
    t = camera_pose.t

    # Transform to camera coordinates
    points_cam = (R @ points_3d.T + t).T

    # Filter points behind camera
    valid_z = points_cam[:, 2] > 0
    points_cam = points_cam[valid_z]

    if len(points_cam) == 0:
        return np.array([]).reshape(0, 2)

    # Project to image plane
    points_proj = (K @ points_cam.T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]

    # Filter to image bounds
    h, w = image_shape
    valid_bounds = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    )

    return points_2d[valid_bounds]


def grow_mask_from_seeds_superpixel(
    image: np.ndarray,
    seed_points: np.ndarray,
    n_segments: int = 500,
) -> np.ndarray:
    """
    Generate segmentation mask by growing from seed points using superpixels.

    Args:
        image: BGR image
        seed_points: Nx2 array of seed point coordinates
        n_segments: Number of SLIC superpixels

    Returns:
        Binary mask
    """
    if len(seed_points) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    # Convert to LAB for better SLIC results
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Create SLIC superpixels
    slic = cv2.ximgproc.createSuperpixelSLIC(lab, cv2.ximgproc.SLIC, n_segments, 10.0)
    slic.iterate(10)
    labels = slic.getLabels()

    # Find superpixels containing seed points
    seed_labels = set()
    for pt in seed_points.astype(int):
        x, y = pt[0], pt[1]
        if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
            seed_labels.add(labels[y, x])

    # Create mask from selected superpixels
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for label in seed_labels:
        mask[labels == label] = 255

    return mask


def grow_mask_from_seeds_floodfill(
    image: np.ndarray,
    seed_points: np.ndarray,
    tolerance: int = 20,
) -> np.ndarray:
    """
    Generate segmentation mask by flood-filling from seed points.

    Args:
        image: BGR image
        seed_points: Nx2 array of seed point coordinates
        tolerance: Color tolerance for flood fill

    Returns:
        Binary mask
    """
    if len(seed_points) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Sample subset of seed points to avoid over-flooding
    if len(seed_points) > 100:
        indices = np.random.choice(len(seed_points), 100, replace=False)
        seed_points = seed_points[indices]

    for pt in seed_points.astype(int):
        x, y = pt[0], pt[1]
        if 0 <= x < w and 0 <= y < h:
            cv2.floodFill(
                image.copy(), mask,
                (x, y),
                newVal=(255, 255, 255),
                loDiff=(tolerance,) * 3,
                upDiff=(tolerance,) * 3,
                flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
            )

    # Extract mask (remove border)
    return (mask[1:-1, 1:-1] * 255).astype(np.uint8)


def grow_mask_from_seeds(
    image: np.ndarray,
    seed_points: np.ndarray,
    method: str = "superpixel",
) -> np.ndarray:
    """
    Generate segmentation mask by growing from seed points.

    Args:
        image: BGR image
        seed_points: Nx2 array of seed coordinates
        method: "superpixel" or "flood_fill"

    Returns:
        Binary mask
    """
    if method == "superpixel":
        try:
            return grow_mask_from_seeds_superpixel(image, seed_points)
        except (cv2.error, AttributeError):
            # Fall back to flood fill if ximgproc not available
            return grow_mask_from_seeds_floodfill(image, seed_points)
    else:
        return grow_mask_from_seeds_floodfill(image, seed_points)


def compute_mask_metrics(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> dict:
    """
    Compute IoU, precision, recall between predicted and ground truth masks.

    Args:
        predicted_mask: Predicted binary mask
        ground_truth_mask: Ground truth binary mask

    Returns:
        Dict with IoU, precision, recall
    """
    pred = predicted_mask > 127
    gt = ground_truth_mask > 127

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    iou = intersection / union if union > 0 else 0.0
    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    recall = intersection / gt_sum if gt_sum > 0 else 0.0

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0,
        "intersection": int(intersection),
        "union": int(union),
        "predicted_area": int(pred_sum),
        "ground_truth_area": int(gt_sum),
    }


def visualize_projected_points(
    image: np.ndarray,
    points_2d: np.ndarray,
    point_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Visualize projected 3D points on image.

    Args:
        image: BGR image
        points_2d: Nx2 array of 2D points
        point_color: Color for points

    Returns:
        Annotated image
    """
    vis = image.copy()

    for pt in points_2d.astype(int):
        cv2.circle(vis, (pt[0], pt[1]), 3, point_color, -1)

    cv2.putText(vis, f"Projected points: {len(points_2d)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return vis


def visualize_mask_diff(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> np.ndarray:
    """
    Create diff visualization showing mask disagreements.

    Colors:
        - Green: Both agree (true positive)
        - Red: SAM only (false negative)
        - Blue: Match-based only (false positive)

    Args:
        predicted_mask: Match-based mask
        ground_truth_mask: SAM mask

    Returns:
        Colored diff image
    """
    pred = predicted_mask > 127
    gt = ground_truth_mask > 127

    diff = np.zeros((*pred.shape, 3), dtype=np.uint8)

    # True positive (both)
    both = np.logical_and(pred, gt)
    diff[both] = (0, 255, 0)  # Green

    # False negative (SAM only)
    sam_only = np.logical_and(gt, ~pred)
    diff[sam_only] = (0, 0, 255)  # Red

    # False positive (match-based only)
    match_only = np.logical_and(pred, ~gt)
    diff[match_only] = (255, 0, 0)  # Blue

    return diff


def create_iou_per_image_plot(
    metrics: list[dict],
) -> np.ndarray:
    """
    Create bar chart of IoU scores per image.

    Args:
        metrics: List of metric dicts per image

    Returns:
        Plot image
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        ious = [m["iou"] for m in metrics]
        x = list(range(len(ious)))

        # Color by IoU (green=good, red=bad)
        colors = [(1 - iou, iou, 0.2) for iou in ious]

        bars = ax.bar(x, ious, color=colors, alpha=0.7)

        # Mean line
        mean_iou = np.mean(ious)
        ax.axhline(mean_iou, color='blue', linestyle='--', linewidth=2,
                  label=f'Mean IoU: {mean_iou:.3f}')

        # Threshold
        ax.axhline(0.5, color='orange', linestyle=':', linewidth=2,
                  label='Warning threshold (0.5)')

        ax.set_xlabel('Image Index', fontsize=12)
        ax.set_ylabel('IoU Score', fontsize=12)
        ax.set_title('Match-Based vs SAM Segmentation IoU', fontsize=14)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    except ImportError:
        img = np.full((400, 800, 3), 255, dtype=np.uint8)
        cv2.putText(img, "matplotlib required for plots", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img


def create_point_coverage_heatmap(
    image_shape: tuple[int, int],
    points_2d: np.ndarray,
    cell_size: int = 32,
) -> np.ndarray:
    """
    Create heatmap of projected point density.

    Args:
        image_shape: (height, width)
        points_2d: Projected 2D points
        cell_size: Heatmap cell size

    Returns:
        Heatmap image
    """
    h, w = image_shape
    grid_h = (h + cell_size - 1) // cell_size
    grid_w = (w + cell_size - 1) // cell_size

    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)

    for pt in points_2d.astype(int):
        gx = min(pt[0] // cell_size, grid_w - 1)
        gy = min(pt[1] // cell_size, grid_h - 1)
        if 0 <= gx < grid_w and 0 <= gy < grid_h:
            heatmap[gy, gx] += 1

    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Apply colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Resize
    return cv2.resize(heatmap_color, (w, h), interpolation=cv2.INTER_NEAREST)


def debug_object_verification(
    ctx: "DebugContext",
    images: list[np.ndarray],
    sam_masks: list[np.ndarray],
    sparse_cloud: o3d.geometry.PointCloud,
    dense_cloud: o3d.geometry.PointCloud,
    camera_poses: list["CameraPose"],
    intrinsics: "CameraIntrinsics",
) -> dict:
    """
    Verify object identification by comparing match-based vs SAM segmentation.

    Outputs:
        - projected_points_XXX.jpg: Images with projected 3D points
        - match_mask_XXX.jpg: Segmentation from matches only
        - sam_mask_XXX.jpg: SAM ground truth
        - mask_diff_XXX.jpg: Difference visualization
        - iou_per_image.png: IoU bar chart
        - verification_metrics.json: Metrics summary
        - point_coverage_XXX.png: Point density heatmaps

    Args:
        ctx: Debug context
        images: Input images
        sam_masks: SAM segmentation masks
        sparse_cloud: Sparse point cloud from SfM
        dense_cloud: Dense point cloud
        camera_poses: Camera poses
        intrinsics: Camera intrinsics

    Returns:
        Dict with per-image and aggregate metrics
    """
    if not ctx.enabled:
        return {}

    # Use dense cloud if available, otherwise sparse
    if len(dense_cloud.points) > 0:
        points_3d = np.asarray(dense_cloud.points)
    else:
        points_3d = np.asarray(sparse_cloud.points) if len(sparse_cloud.points) > 0 else np.array([])

    all_metrics = []

    for i, (image, sam_mask) in enumerate(zip(images, sam_masks)):
        # Find matching camera pose
        pose = None
        for p in camera_poses:
            if p.image_idx == i:
                pose = p
                break

        if pose is None:
            # Use identity pose as fallback
            from app.pipeline.sfm import CameraPose
            pose = CameraPose(R=np.eye(3), t=np.zeros((3, 1)), image_idx=i)

        # Project 3D points
        points_2d = project_points_to_image(points_3d, pose, intrinsics, image.shape[:2])

        # Visualize projected points
        proj_vis = visualize_projected_points(image, points_2d)
        ctx.save_image("object_verification", f"projected_points_{i:03d}", proj_vis)

        # Point coverage heatmap
        heatmap = create_point_coverage_heatmap(image.shape[:2], points_2d)
        ctx.save_image("object_verification", f"point_coverage_{i:03d}.png", heatmap)

        # Grow mask from projected points
        match_mask = grow_mask_from_seeds(image, points_2d, method="superpixel")
        ctx.save_image("object_verification", f"match_mask_{i:03d}", match_mask)

        # Save SAM mask for comparison
        sam_vis = cv2.cvtColor(sam_mask, cv2.COLOR_GRAY2BGR)
        ctx.save_image("object_verification", f"sam_mask_{i:03d}", sam_vis)

        # Create diff visualization
        diff = visualize_mask_diff(match_mask, sam_mask)
        ctx.save_image("object_verification", f"mask_diff_{i:03d}", diff)

        # Compute metrics
        metrics = compute_mask_metrics(match_mask, sam_mask)
        metrics["image_idx"] = i
        metrics["num_projected_points"] = len(points_2d)
        all_metrics.append(metrics)

    # Create IoU plot
    iou_plot = create_iou_per_image_plot(all_metrics)
    ctx.save_image("object_verification", "iou_per_image.png", iou_plot)

    # Compute aggregate metrics
    ious = [m["iou"] for m in all_metrics]
    aggregate = {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "median_iou": float(np.median(ious)) if ious else 0.0,
        "min_iou": float(np.min(ious)) if ious else 0.0,
        "max_iou": float(np.max(ious)) if ious else 0.0,
        "std_iou": float(np.std(ious)) if ious else 0.0,
        "num_images": len(all_metrics),
        "low_iou_images": [m["image_idx"] for m in all_metrics if m["iou"] < 0.5],
    }

    # Save metrics
    results = {
        "aggregate": aggregate,
        "per_image": all_metrics,
        "point_cloud_used": "dense" if len(dense_cloud.points) > 0 else "sparse",
        "total_3d_points": len(points_3d),
    }
    ctx.save_json("object_verification", "verification_metrics", results)

    return results
