"""Debug visualizations for the Structure from Motion stage."""

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from app.pipeline.debug import DebugContext
    from app.pipeline.sfm import CameraPose


def render_point_cloud_view(
    cloud: o3d.geometry.PointCloud,
    camera_params: dict,
    width: int = 800,
    height: int = 600,
) -> np.ndarray:
    """
    Render point cloud from a specific viewpoint.

    Args:
        cloud: Point cloud to render
        camera_params: Camera parameters (front, lookat, up, zoom)
        width, height: Output image dimensions

    Returns:
        Rendered image
    """
    if len(cloud.points) == 0:
        img = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.putText(img, "Empty point cloud", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(cloud)

    # Set viewpoint
    ctr = vis.get_view_control()
    ctr.set_front(camera_params.get("front", [0, 0, -1]))
    ctr.set_lookat(camera_params.get("lookat", [0, 0, 0]))
    ctr.set_up(camera_params.get("up", [0, -1, 0]))
    ctr.set_zoom(camera_params.get("zoom", 0.8))

    # Render
    vis.poll_events()
    vis.update_renderer()

    # Capture
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    # Convert to uint8 BGR
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def create_camera_poses_plot(
    poses: list["CameraPose"],
    cloud: Optional[o3d.geometry.PointCloud] = None,
) -> np.ndarray:
    """
    Create 3D visualization of camera poses.

    Args:
        poses: List of camera poses
        cloud: Optional point cloud for context

    Returns:
        Plot image
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot camera positions
        positions = []
        for pose in poses:
            # Camera center in world coordinates: C = -R^T * t
            C = -pose.R.T @ pose.t
            positions.append(C.flatten())

            # Draw camera as a small frustum
            C = C.flatten()
            ax.scatter(C[0], C[1], C[2], c='red', s=100, marker='^')

            # Draw optical axis
            axis_length = 0.2
            optical_axis = pose.R.T @ np.array([[0], [0], [1]]) * axis_length
            ax.quiver(C[0], C[1], C[2],
                     optical_axis[0, 0], optical_axis[1, 0], optical_axis[2, 0],
                     color='blue', arrow_length_ratio=0.3)

            # Label
            ax.text(C[0], C[1], C[2], f'{pose.image_idx}', fontsize=8)

        # Plot point cloud sample if available
        if cloud and len(cloud.points) > 0:
            points = np.asarray(cloud.points)
            # Sample points
            if len(points) > 1000:
                indices = np.random.choice(len(points), 1000, replace=False)
                points = points[indices]

            colors = np.asarray(cloud.colors) if cloud.has_colors() else None
            if colors is not None and len(colors) == len(np.asarray(cloud.points)):
                colors = colors[indices] if len(points) < len(np.asarray(cloud.points)) else colors[:1000]
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c=colors, s=1, alpha=0.5)
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c='gray', s=1, alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Poses and Point Cloud')

        # Make axes equal
        if positions:
            positions = np.array(positions)
            max_range = np.ptp(positions, axis=0).max() / 2
            mid = positions.mean(axis=0)
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        fig.tight_layout()

        # Convert to image
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    except ImportError:
        img = np.full((600, 800, 3), 255, dtype=np.uint8)
        cv2.putText(img, "matplotlib required for plots", (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img


def create_reprojection_error_plot(
    errors: list[float],
) -> np.ndarray:
    """
    Create histogram of reprojection errors.

    Args:
        errors: List of reprojection errors

    Returns:
        Plot image
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(errors), color='red', linestyle='--',
                   label=f'Mean: {np.mean(errors):.2f}px')
        ax1.axvline(np.median(errors), color='green', linestyle='--',
                   label=f'Median: {np.median(errors):.2f}px')
        ax1.set_xlabel('Reprojection Error (pixels)', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Reprojection Error Distribution', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Sorted errors plot
        sorted_errors = np.sort(errors)
        ax2.plot(sorted_errors, 'b-', linewidth=0.5)
        ax2.axhline(np.mean(errors), color='red', linestyle='--', label='Mean')
        ax2.fill_between(range(len(sorted_errors)), sorted_errors, alpha=0.3)
        ax2.set_xlabel('Point Index (sorted)', fontsize=12)
        ax2.set_ylabel('Reprojection Error (pixels)', fontsize=12)
        ax2.set_title('Sorted Reprojection Errors', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()

        # Convert to image
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


def debug_sfm(
    ctx: "DebugContext",
    sparse_cloud: o3d.geometry.PointCloud,
    camera_poses: list["CameraPose"],
    reprojection_errors: Optional[list[float]] = None,
    triangulation_stats: Optional[dict] = None,
) -> None:
    """
    Generate debug visualizations for the SfM stage.

    Outputs:
        - sparse_cloud.ply: Sparse point cloud file
        - sparse_cloud_top.png: Top-down view
        - sparse_cloud_side.png: Side view
        - camera_poses.png: 3D plot with camera frustums
        - camera_poses.json: Camera matrices
        - reprojection_errors.png: Error histogram
        - triangulation_stats.json: Reconstruction statistics

    Args:
        ctx: Debug context
        sparse_cloud: Sparse 3D point cloud
        camera_poses: List of camera poses
        reprojection_errors: Per-point reprojection errors
        triangulation_stats: Triangulation statistics
    """
    if not ctx.enabled:
        return

    # Save point cloud
    ctx.save_point_cloud("sfm", "sparse_cloud", sparse_cloud)

    # Render views
    if len(sparse_cloud.points) > 0:
        # Top view
        top_view = render_point_cloud_view(sparse_cloud, {
            "front": [0, -1, 0],
            "lookat": [0, 0, 0],
            "up": [0, 0, -1],
            "zoom": 0.6
        })
        ctx.save_image("sfm", "sparse_cloud_top.png", top_view)

        # Side view
        side_view = render_point_cloud_view(sparse_cloud, {
            "front": [1, 0, 0],
            "lookat": [0, 0, 0],
            "up": [0, -1, 0],
            "zoom": 0.6
        })
        ctx.save_image("sfm", "sparse_cloud_side.png", side_view)

    # Camera poses visualization
    poses_plot = create_camera_poses_plot(camera_poses, sparse_cloud)
    ctx.save_image("sfm", "camera_poses.png", poses_plot)

    # Save camera poses as JSON
    poses_data = {
        "num_cameras": len(camera_poses),
        "cameras": [
            {
                "image_idx": pose.image_idx,
                "R": pose.R.tolist(),
                "t": pose.t.tolist(),
                "center": (-pose.R.T @ pose.t).flatten().tolist(),
            }
            for pose in camera_poses
        ]
    }
    ctx.save_json("sfm", "camera_poses", poses_data)

    # Reprojection errors
    if reprojection_errors:
        error_plot = create_reprojection_error_plot(reprojection_errors)
        ctx.save_image("sfm", "reprojection_errors.png", error_plot)

    # Statistics
    stats = {
        "num_3d_points": len(sparse_cloud.points),
        "num_cameras": len(camera_poses),
        "reprojection_errors": {
            "mean": float(np.mean(reprojection_errors)) if reprojection_errors else None,
            "median": float(np.median(reprojection_errors)) if reprojection_errors else None,
            "std": float(np.std(reprojection_errors)) if reprojection_errors else None,
            "max": float(np.max(reprojection_errors)) if reprojection_errors else None,
        },
        "triangulation": triangulation_stats or {},
    }
    ctx.save_json("sfm", "triangulation_stats", stats)
