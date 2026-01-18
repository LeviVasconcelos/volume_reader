"""Debug visualizations for the volume calculation stage."""

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from app.pipeline.debug import DebugContext


def render_voxel_grid(
    voxel_grid: o3d.geometry.VoxelGrid,
    width: int = 800,
    height: int = 600,
) -> np.ndarray:
    """
    Render voxel grid to image.

    Args:
        voxel_grid: Voxel grid to render
        width, height: Output dimensions

    Returns:
        Rendered image
    """
    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        img = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.putText(img, "Empty voxel grid", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])

    vis.add_geometry(voxel_grid)
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def create_voxel_slice(
    voxel_grid: o3d.geometry.VoxelGrid,
    axis: str,
    slice_idx: Optional[int] = None,
    size: int = 400,
) -> np.ndarray:
    """
    Create 2D slice through voxel grid.

    Args:
        voxel_grid: Voxel grid
        axis: Axis perpendicular to slice ("xy", "xz", "yz")
        slice_idx: Index of slice (None = middle)
        size: Output image size

    Returns:
        Slice visualization
    """
    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        img = np.full((size, size, 3), 40, dtype=np.uint8)
        cv2.putText(img, f"Empty ({axis})", (50, size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    # Get voxel indices
    indices = np.array([v.grid_index for v in voxels])

    # Determine axis mapping
    axis_map = {
        "xy": (0, 1, 2),  # X, Y, Z (slice along Z)
        "xz": (0, 2, 1),  # X, Z, Y (slice along Y)
        "yz": (1, 2, 0),  # Y, Z, X (slice along X)
    }
    ax1, ax2, slice_ax = axis_map.get(axis, (0, 1, 2))

    # Get slice index
    slice_range = indices[:, slice_ax]
    if slice_idx is None:
        slice_idx = (slice_range.min() + slice_range.max()) // 2

    # Filter voxels in slice
    slice_mask = indices[:, slice_ax] == slice_idx
    slice_voxels = indices[slice_mask]

    # Create image
    if len(slice_voxels) == 0:
        img = np.full((size, size, 3), 40, dtype=np.uint8)
        cv2.putText(img, f"Empty slice ({axis})", (50, size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    # Map to 2D
    x_coords = slice_voxels[:, ax1]
    y_coords = slice_voxels[:, ax2]

    # Normalize to image coordinates
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    margin = 20
    img_size = size - 2 * margin

    if x_max > x_min:
        x_norm = (x_coords - x_min) / (x_max - x_min) * img_size + margin
    else:
        x_norm = np.full_like(x_coords, size // 2, dtype=float)

    if y_max > y_min:
        y_norm = (y_coords - y_min) / (y_max - y_min) * img_size + margin
    else:
        y_norm = np.full_like(y_coords, size // 2, dtype=float)

    # Draw voxels as filled squares
    img = np.full((size, size, 3), 40, dtype=np.uint8)

    voxel_size_px = max(3, img_size // max(x_max - x_min + 1, y_max - y_min + 1, 1))

    for x, y in zip(x_norm.astype(int), y_norm.astype(int)):
        cv2.rectangle(
            img,
            (x - voxel_size_px // 2, y - voxel_size_px // 2),
            (x + voxel_size_px // 2, y + voxel_size_px // 2),
            (0, 200, 100),
            -1
        )

    # Add label
    cv2.putText(img, f"{axis.upper()} slice @ {slice_idx}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"Voxels: {len(slice_voxels)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return img


def visualize_bounding_box(
    mesh: o3d.geometry.TriangleMesh,
    width: int = 800,
    height: int = 600,
) -> np.ndarray:
    """
    Visualize mesh with oriented bounding box.

    Args:
        mesh: Triangle mesh
        width, height: Output dimensions

    Returns:
        Rendered image
    """
    if len(mesh.triangles) == 0:
        img = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.putText(img, "Empty mesh", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    # Get oriented bounding box
    obb = mesh.get_oriented_bounding_box()
    obb.color = (1, 0, 0)  # Red

    # Also get axis-aligned for reference
    aabb = mesh.get_axis_aligned_bounding_box()
    aabb.color = (0, 1, 0)  # Green

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])

    mesh_vis = o3d.geometry.TriangleMesh(mesh)
    mesh_vis.paint_uniform_color([0.6, 0.6, 0.6])
    vis.add_geometry(mesh_vis)
    vis.add_geometry(obb)
    vis.add_geometry(aabb)

    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Add legend
    cv2.putText(img, "Red: Oriented BB", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
    cv2.putText(img, "Green: Axis-aligned BB", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    return img


def create_dimensions_annotated(
    mesh: o3d.geometry.TriangleMesh,
    dimensions: tuple[float, float, float],
    scale_factor: float,
) -> np.ndarray:
    """
    Create plot showing mesh dimensions.

    Args:
        mesh: Triangle mesh
        dimensions: (length_mm, width_mm, thickness_mm)
        scale_factor: Scale factor used

    Returns:
        Annotated plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        length, width, thickness = dimensions

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart of dimensions
        labels = ['Length', 'Width', 'Thickness']
        values = [length, width, thickness]
        colors = ['steelblue', 'green', 'coral']

        bars = ax1.bar(labels, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Dimension (mm)', fontsize=12)
        ax1.set_title('Board Dimensions', fontsize=14)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.annotate(f'{val:.1f} mm',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)

        # Schematic side view
        ax2.set_xlim(-0.1, 1.1)
        ax2.set_ylim(-0.1, 1.1)

        # Draw schematic rectangle (scaled for visualization)
        max_dim = max(length, width, thickness)
        l_norm = length / max_dim * 0.8
        w_norm = width / max_dim * 0.8
        t_norm = thickness / max_dim * 0.8

        # Top view rectangle
        rect = plt.Rectangle((0.1, 0.3), l_norm, w_norm, fill=False,
                             edgecolor='steelblue', linewidth=2)
        ax2.add_patch(rect)

        # Dimension annotations
        ax2.annotate('', xy=(0.1 + l_norm, 0.25), xytext=(0.1, 0.25),
                    arrowprops=dict(arrowstyle='<->', color='black'))
        ax2.text(0.1 + l_norm / 2, 0.18, f'{length:.0f} mm',
                ha='center', fontsize=10)

        ax2.annotate('', xy=(0.05, 0.3), xytext=(0.05, 0.3 + w_norm),
                    arrowprops=dict(arrowstyle='<->', color='black'))
        ax2.text(0.02, 0.3 + w_norm / 2, f'{width:.0f} mm',
                ha='center', va='center', rotation=90, fontsize=10)

        ax2.set_title('Schematic View (Top)', fontsize=14)
        ax2.set_aspect('equal')
        ax2.axis('off')

        # Add scale info
        fig.suptitle(f'Scale Factor: {scale_factor:.4f} mm/unit', fontsize=11, y=0.02)

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


def debug_volume(
    ctx: "DebugContext",
    mesh: o3d.geometry.TriangleMesh,
    voxel_grid: o3d.geometry.VoxelGrid,
    voxel_size: float,
    dimensions: tuple[float, float, float],
    scale_factor: float,
    volume_liters: float,
) -> None:
    """
    Generate debug visualizations for the volume calculation stage.

    Outputs:
        - voxel_grid.ply: Voxelized mesh (as point cloud)
        - voxel_slice_xy.png, voxel_slice_xz.png, voxel_slice_yz.png: Slices
        - bounding_box.png: Mesh with bounding boxes
        - dimensions_annotated.png: Dimension labels
        - volume_breakdown.json: Calculation breakdown

    Args:
        ctx: Debug context
        mesh: Triangle mesh
        voxel_grid: Voxel grid used for volume
        voxel_size: Voxel size in mesh units
        dimensions: (length_mm, width_mm, thickness_mm)
        scale_factor: Scale conversion factor
        volume_liters: Computed volume
    """
    if not ctx.enabled:
        return

    # Save voxel grid as point cloud (for visualization)
    voxels = voxel_grid.get_voxels()
    if len(voxels) > 0:
        centers = np.array([voxel_grid.get_voxel_center_coordinate(v.grid_index) for v in voxels])
        voxel_cloud = o3d.geometry.PointCloud()
        voxel_cloud.points = o3d.utility.Vector3dVector(centers)
        ctx.save_point_cloud("volume", "voxel_grid", voxel_cloud)

    # Render voxel grid
    voxel_render = render_voxel_grid(voxel_grid)
    ctx.save_image("volume", "voxel_grid_view.png", voxel_render)

    # Create slices
    for axis in ["xy", "xz", "yz"]:
        slice_img = create_voxel_slice(voxel_grid, axis)
        ctx.save_image("volume", f"voxel_slice_{axis}.png", slice_img)

    # Bounding box visualization
    bbox_vis = visualize_bounding_box(mesh)
    ctx.save_image("volume", "bounding_box.png", bbox_vis)

    # Dimensions annotated
    dims_vis = create_dimensions_annotated(mesh, dimensions, scale_factor)
    ctx.save_image("volume", "dimensions_annotated.png", dims_vis)

    # Volume breakdown
    n_voxels = len(voxels)
    voxel_volume = voxel_size ** 3
    raw_volume = n_voxels * voxel_volume
    scaled_volume_mm3 = raw_volume * (scale_factor ** 3)
    volume_liters_calc = scaled_volume_mm3 / 1_000_000

    breakdown = {
        "voxel_size_mesh_units": voxel_size,
        "voxel_size_mm": voxel_size * scale_factor,
        "num_voxels": n_voxels,
        "voxel_volume_mesh_units": voxel_volume,
        "raw_volume_mesh_units": raw_volume,
        "scale_factor": scale_factor,
        "scaled_volume_mm3": scaled_volume_mm3,
        "volume_liters": volume_liters,
        "dimensions": {
            "length_mm": dimensions[0],
            "width_mm": dimensions[1],
            "thickness_mm": dimensions[2],
        },
        "oriented_bounding_box": {
            "extent": mesh.get_oriented_bounding_box().extent.tolist() if len(mesh.triangles) > 0 else None,
        },
    }
    ctx.save_json("volume", "volume_breakdown", breakdown)
