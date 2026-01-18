"""Debug visualizations for the dense reconstruction stage."""

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from app.pipeline.debug import DebugContext


def colorize_depth_map(
    depth_map: np.ndarray,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
) -> np.ndarray:
    """
    Convert depth map to pseudocolor visualization.

    Args:
        depth_map: Depth values (0 = invalid)
        min_depth: Minimum depth for coloring
        max_depth: Maximum depth for coloring

    Returns:
        Colored depth map (BGR)
    """
    valid_mask = depth_map > 0

    if not valid_mask.any():
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)

    # Determine range
    valid_depths = depth_map[valid_mask]
    if min_depth is None:
        min_depth = np.percentile(valid_depths, 5)
    if max_depth is None:
        max_depth = np.percentile(valid_depths, 95)

    # Normalize
    depth_norm = np.clip((depth_map - min_depth) / (max_depth - min_depth + 1e-8), 0, 1)
    depth_norm[~valid_mask] = 0

    # Apply colormap
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

    # Mark invalid regions as black
    colored[~valid_mask] = 0

    return colored


def create_depth_validity_mask(depth_map: np.ndarray) -> np.ndarray:
    """
    Create visualization of valid depth regions.

    Args:
        depth_map: Depth values

    Returns:
        Binary mask visualization
    """
    valid = (depth_map > 0).astype(np.uint8) * 255
    # Convert to 3-channel
    valid_vis = cv2.cvtColor(valid, cv2.COLOR_GRAY2BGR)

    # Add statistics
    valid_pct = (depth_map > 0).sum() / depth_map.size * 100
    cv2.putText(valid_vis, f"Valid: {valid_pct:.1f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return valid_vis


def create_stereo_rectified_visualization(
    img1: np.ndarray,
    img2: np.ndarray,
    num_lines: int = 20,
) -> np.ndarray:
    """
    Create visualization of stereo rectified images with horizontal lines.

    Args:
        img1, img2: Rectified images
        num_lines: Number of horizontal guide lines

    Returns:
        Combined visualization
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Resize to match height
    if h1 != h2:
        scale = h1 / h2
        img2 = cv2.resize(img2, (int(w2 * scale), h1))

    combined = np.hstack([img1, img2])

    # Draw horizontal lines
    h, w = combined.shape[:2]
    for i in range(num_lines):
        y = int(h * i / num_lines)
        color = ((i * 37) % 256, (i * 73) % 256, (i * 127) % 256)
        cv2.line(combined, (0, y), (w, y), color, 1)

    return combined


def render_point_cloud(
    cloud: o3d.geometry.PointCloud,
    width: int = 800,
    height: int = 600,
    background_color: tuple = (0.1, 0.1, 0.1),
) -> np.ndarray:
    """
    Render point cloud to image.

    Args:
        cloud: Point cloud to render
        width, height: Output dimensions
        background_color: Background color

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

    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)
    opt.point_size = 2.0

    vis.add_geometry(cloud)

    # Auto-center view
    vis.reset_view_point(True)

    vis.poll_events()
    vis.update_renderer()

    # Capture
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def visualize_normals(
    cloud: o3d.geometry.PointCloud,
    width: int = 800,
    height: int = 600,
    normal_length: float = 0.02,
) -> np.ndarray:
    """
    Visualize point cloud with normal vectors.

    Args:
        cloud: Point cloud with normals
        width, height: Output dimensions
        normal_length: Length of normal arrows

    Returns:
        Rendered image
    """
    if len(cloud.points) == 0 or not cloud.has_normals():
        img = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.putText(img, "No normals available", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    # Sample points for visualization
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)

    if len(points) > 5000:
        indices = np.random.choice(len(points), 5000, replace=False)
        points = points[indices]
        normals = normals[indices]

    # Create line set for normals
    line_points = []
    line_indices = []
    for i, (p, n) in enumerate(zip(points, normals)):
        line_points.append(p)
        line_points.append(p + n * normal_length)
        line_indices.append([i * 2, i * 2 + 1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_indices)
    line_set.paint_uniform_color([1, 0, 0])  # Red normals

    # Render
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(cloud)
    vis.add_geometry(line_set)
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def create_outlier_removal_comparison(
    before_count: int,
    after_count: int,
) -> np.ndarray:
    """
    Create visualization comparing before/after outlier removal.

    Args:
        before_count: Points before removal
        after_count: Points after removal

    Returns:
        Comparison plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(['Before', 'After'], [before_count, after_count],
                     color=['steelblue', 'green'], alpha=0.7)

        removed = before_count - after_count
        removed_pct = removed / before_count * 100 if before_count > 0 else 0

        ax.set_ylabel('Number of Points', fontsize=12)
        ax.set_title(f'Outlier Removal: {removed:,} points removed ({removed_pct:.1f}%)', fontsize=14)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height):,}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=12)

        fig.tight_layout()

        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    except ImportError:
        img = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.putText(img, "matplotlib required for plots", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img


def debug_dense(
    ctx: "DebugContext",
    depth_maps: list[tuple[int, int, np.ndarray]],
    dense_cloud: o3d.geometry.PointCloud,
    intermediate_clouds: Optional[list[o3d.geometry.PointCloud]] = None,
    outlier_stats: Optional[dict] = None,
) -> None:
    """
    Generate debug visualizations for the dense reconstruction stage.

    Outputs:
        - depth_map_I_J.png: Pseudocolor depth maps
        - depth_validity_I_J.png: Valid depth regions
        - dense_cloud.ply: Full dense point cloud
        - dense_cloud_view.png: Rendered view
        - outlier_removal.png: Before/after comparison
        - normals_visualization.png: Point cloud with normals
        - dense_stats.json: Statistics

    Args:
        ctx: Debug context
        depth_maps: List of (i, j, depth_map) tuples
        dense_cloud: Final dense point cloud
        intermediate_clouds: Per-pair clouds before merging
        outlier_stats: Outlier removal statistics
    """
    if not ctx.enabled:
        return

    # Save depth maps
    for i, j, depth_map in depth_maps:
        # Colored depth
        colored = colorize_depth_map(depth_map)
        ctx.save_image("dense", f"depth_map_{i}_{j}.png", colored)

        # Validity mask
        validity = create_depth_validity_mask(depth_map)
        ctx.save_image("dense", f"depth_validity_{i}_{j}.png", validity)

    # Save dense cloud
    ctx.save_point_cloud("dense", "dense_cloud", dense_cloud)

    # Render view
    view = render_point_cloud(dense_cloud)
    ctx.save_image("dense", "dense_cloud_view.png", view)

    # Normals visualization
    if dense_cloud.has_normals():
        normals_vis = visualize_normals(dense_cloud)
        ctx.save_image("dense", "normals_visualization.png", normals_vis)

    # Outlier removal comparison
    if outlier_stats:
        comparison = create_outlier_removal_comparison(
            outlier_stats.get("before", 0),
            outlier_stats.get("after", 0)
        )
        ctx.save_image("dense", "outlier_removal.png", comparison)

    # Statistics
    stats = {
        "num_depth_maps": len(depth_maps),
        "depth_map_pairs": [
            {
                "i": i, "j": j,
                "valid_pixels": int((dm > 0).sum()),
                "valid_percent": float((dm > 0).sum() / dm.size * 100),
                "min_depth": float(dm[dm > 0].min()) if (dm > 0).any() else None,
                "max_depth": float(dm[dm > 0].max()) if (dm > 0).any() else None,
            }
            for i, j, dm in depth_maps
        ],
        "dense_cloud": {
            "num_points": len(dense_cloud.points),
            "has_colors": dense_cloud.has_colors(),
            "has_normals": dense_cloud.has_normals(),
        },
        "outlier_removal": outlier_stats or {},
    }
    ctx.save_json("dense", "dense_stats", stats)
