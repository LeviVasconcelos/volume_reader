"""Debug visualizations for the mesh generation stage."""

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from app.pipeline.debug import DebugContext


def render_mesh(
    mesh: o3d.geometry.TriangleMesh,
    width: int = 800,
    height: int = 600,
    wireframe: bool = False,
    background_color: tuple = (0.1, 0.1, 0.1),
) -> np.ndarray:
    """
    Render mesh to image.

    Args:
        mesh: Triangle mesh to render
        width, height: Output dimensions
        wireframe: If True, render as wireframe
        background_color: Background color

    Returns:
        Rendered image
    """
    if len(mesh.triangles) == 0:
        img = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.putText(img, "Empty mesh", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)
    opt.mesh_show_wireframe = wireframe
    opt.mesh_show_back_face = True

    if wireframe:
        # Create wireframe from mesh edges
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        if len(edges) == 0:
            # Create line set from triangles
            triangles = np.asarray(mesh.triangles)
            vertices = np.asarray(mesh.vertices)
            lines = []
            for tri in triangles:
                lines.append([tri[0], tri[1]])
                lines.append([tri[1], tri[2]])
                lines.append([tri[2], tri[0]])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0.8, 0.8, 0.8])
            vis.add_geometry(line_set)
        else:
            vis.add_geometry(mesh)
    else:
        # Ensure normals for proper shading
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        vis.add_geometry(mesh)

    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def visualize_vertex_density(
    mesh: o3d.geometry.TriangleMesh,
    densities: np.ndarray,
    width: int = 800,
    height: int = 600,
) -> np.ndarray:
    """
    Visualize mesh vertices colored by Poisson density.

    Args:
        mesh: Triangle mesh
        densities: Per-vertex density values
        width, height: Output dimensions

    Returns:
        Rendered image
    """
    if len(mesh.triangles) == 0 or len(densities) == 0:
        img = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.putText(img, "No density data", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    # Create colored mesh
    mesh_colored = o3d.geometry.TriangleMesh(mesh)

    # Normalize densities
    densities = np.asarray(densities)
    min_d, max_d = densities.min(), densities.max()
    if max_d > min_d:
        norm_densities = (densities - min_d) / (max_d - min_d)
    else:
        norm_densities = np.ones_like(densities) * 0.5

    # Apply colormap (low = blue, high = red)
    colors = np.zeros((len(norm_densities), 3))
    colors[:, 0] = norm_densities  # R
    colors[:, 2] = 1 - norm_densities  # B

    mesh_colored.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Render
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh_colored)
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Add colorbar legend
    cv2.putText(img, "Density: Low", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
    cv2.putText(img, "High", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

    return img


def visualize_mesh_issues(
    mesh: o3d.geometry.TriangleMesh,
    width: int = 800,
    height: int = 600,
) -> np.ndarray:
    """
    Visualize mesh quality issues (non-manifold edges, degenerate triangles).

    Args:
        mesh: Triangle mesh
        width, height: Output dimensions

    Returns:
        Rendered image with issues highlighted
    """
    if len(mesh.triangles) == 0:
        img = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.putText(img, "Empty mesh", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    # Get non-manifold edges
    non_manifold_edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)

    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    # Add mesh with transparency
    mesh_vis = o3d.geometry.TriangleMesh(mesh)
    mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])
    vis.add_geometry(mesh_vis)

    # Highlight non-manifold edges in red
    if len(non_manifold_edges) > 0:
        vertices = np.asarray(mesh.vertices)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(non_manifold_edges)
        line_set.paint_uniform_color([1, 0, 0])  # Red
        vis.add_geometry(line_set)

    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Add info
    n_non_manifold = len(non_manifold_edges)
    cv2.putText(img, f"Non-manifold edges: {n_non_manifold}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if n_non_manifold > 0 else (0, 255, 0), 2)

    return img


def visualize_mesh_holes(
    mesh: o3d.geometry.TriangleMesh,
    width: int = 800,
    height: int = 600,
) -> np.ndarray:
    """
    Visualize boundary edges (holes) in the mesh.

    Args:
        mesh: Triangle mesh
        width, height: Output dimensions

    Returns:
        Rendered image with holes highlighted
    """
    if len(mesh.triangles) == 0:
        img = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.putText(img, "Empty mesh", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    # Get boundary edges (edges with only one adjacent triangle)
    edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    mesh_vis = o3d.geometry.TriangleMesh(mesh)
    mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])
    vis.add_geometry(mesh_vis)

    # Highlight boundary edges in yellow
    if len(edges) > 0:
        vertices = np.asarray(mesh.vertices)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.paint_uniform_color([1, 1, 0])  # Yellow
        vis.add_geometry(line_set)

    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Add info
    is_watertight = mesh.is_watertight()
    status = "Watertight" if is_watertight else f"Has holes ({len(edges)} boundary edges)"
    color = (0, 255, 0) if is_watertight else (0, 255, 255)
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def create_smoothing_comparison(
    before_mesh: o3d.geometry.TriangleMesh,
    after_mesh: o3d.geometry.TriangleMesh,
    width: int = 800,
    height: int = 400,
) -> np.ndarray:
    """
    Create side-by-side comparison of mesh before/after smoothing.

    Args:
        before_mesh: Mesh before smoothing
        after_mesh: Mesh after smoothing
        width, height: Output dimensions (for each view)

    Returns:
        Combined comparison image
    """
    before_img = render_mesh(before_mesh, width // 2, height)
    after_img = render_mesh(after_mesh, width // 2, height)

    # Add labels
    cv2.putText(before_img, "Before Smoothing", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(after_img, "After Smoothing", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return np.hstack([before_img, after_img])


def debug_mesh(
    ctx: "DebugContext",
    mesh: o3d.geometry.TriangleMesh,
    densities: Optional[np.ndarray] = None,
    is_watertight: Optional[bool] = None,
    method_used: str = "poisson",
    pre_smooth_mesh: Optional[o3d.geometry.TriangleMesh] = None,
) -> None:
    """
    Generate debug visualizations for the mesh generation stage.

    Outputs:
        - mesh.ply: Final mesh file
        - mesh_wireframe.png: Wireframe rendering
        - mesh_shaded.png: Shaded surface rendering
        - vertex_density.png: Vertices colored by density
        - mesh_issues.png: Non-manifold areas highlighted
        - mesh_holes.png: Boundary edges (holes) highlighted
        - smoothing_comparison.png: Before/after smoothing
        - mesh_stats.json: Mesh statistics

    Args:
        ctx: Debug context
        mesh: Final triangle mesh
        densities: Per-vertex Poisson densities
        is_watertight: Whether mesh is watertight
        method_used: Meshing method ("poisson" or "ball_pivoting")
        pre_smooth_mesh: Mesh before smoothing
    """
    if not ctx.enabled:
        return

    # Save mesh file
    ctx.save_mesh("mesh", "mesh", mesh)

    # Render wireframe
    wireframe = render_mesh(mesh, wireframe=True)
    ctx.save_image("mesh", "mesh_wireframe.png", wireframe)

    # Render shaded
    shaded = render_mesh(mesh, wireframe=False)
    ctx.save_image("mesh", "mesh_shaded.png", shaded)

    # Vertex density
    if densities is not None and len(densities) > 0:
        density_vis = visualize_vertex_density(mesh, densities)
        ctx.save_image("mesh", "vertex_density.png", density_vis)

    # Mesh issues
    issues_vis = visualize_mesh_issues(mesh)
    ctx.save_image("mesh", "mesh_issues.png", issues_vis)

    # Mesh holes
    holes_vis = visualize_mesh_holes(mesh)
    ctx.save_image("mesh", "mesh_holes.png", holes_vis)

    # Smoothing comparison
    if pre_smooth_mesh is not None:
        comparison = create_smoothing_comparison(pre_smooth_mesh, mesh)
        ctx.save_image("mesh", "smoothing_comparison.png", comparison)

    # Compute watertight if not provided
    if is_watertight is None:
        is_watertight = mesh.is_watertight()

    # Statistics
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    stats = {
        "method_used": method_used,
        "num_vertices": len(vertices),
        "num_triangles": len(triangles),
        "is_watertight": is_watertight,
        "has_vertex_normals": mesh.has_vertex_normals(),
        "has_vertex_colors": mesh.has_vertex_colors(),
        "bounding_box": {
            "min": vertices.min(axis=0).tolist() if len(vertices) > 0 else None,
            "max": vertices.max(axis=0).tolist() if len(vertices) > 0 else None,
        },
        "density_stats": {
            "min": float(densities.min()) if densities is not None else None,
            "max": float(densities.max()) if densities is not None else None,
            "mean": float(densities.mean()) if densities is not None else None,
        } if densities is not None else None,
    }
    ctx.save_json("mesh", "mesh_stats", stats)
