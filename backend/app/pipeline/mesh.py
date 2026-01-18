import asyncio

import numpy as np
import open3d as o3d


def generate_poisson_mesh(
    point_cloud: o3d.geometry.PointCloud,
    depth: int = 9,
    scale: float = 1.1,
) -> tuple[o3d.geometry.TriangleMesh, list[str]]:
    """
    Generate mesh using Poisson surface reconstruction.

    Args:
        point_cloud: Input point cloud with normals
        depth: Octree depth (higher = more detail but slower)
        scale: Scale factor for the reconstruction

    Returns:
        mesh: Generated triangle mesh
        warnings: List of warnings
    """
    warnings = []

    if len(point_cloud.points) < 100:
        warnings.append("Point cloud too sparse for mesh generation")
        return o3d.geometry.TriangleMesh(), warnings

    # Ensure normals exist
    if not point_cloud.has_normals():
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        point_cloud.orient_normals_consistent_tangent_plane(k=15)

    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=depth,
        scale=scale,
        linear_fit=False,
    )

    # Remove low-density vertices (often noise at boundaries)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh, warnings


def generate_ball_pivoting_mesh(
    point_cloud: o3d.geometry.PointCloud,
) -> tuple[o3d.geometry.TriangleMesh, list[str]]:
    """
    Generate mesh using Ball Pivoting Algorithm.

    Better for preserving sharp features but less robust to noise.

    Args:
        point_cloud: Input point cloud with normals

    Returns:
        mesh: Generated triangle mesh
        warnings: List of warnings
    """
    warnings = []

    if len(point_cloud.points) < 100:
        warnings.append("Point cloud too sparse for mesh generation")
        return o3d.geometry.TriangleMesh(), warnings

    # Ensure normals exist
    if not point_cloud.has_normals():
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )

    # Estimate ball radii based on point cloud density
    distances = point_cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [avg_dist * 0.5, avg_dist, avg_dist * 2, avg_dist * 4]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud,
        o3d.utility.DoubleVector(radii),
    )

    return mesh, warnings


def repair_mesh(
    mesh: o3d.geometry.TriangleMesh,
) -> tuple[o3d.geometry.TriangleMesh, list[str]]:
    """
    Repair mesh to make it watertight.

    Args:
        mesh: Input mesh (potentially with holes)

    Returns:
        mesh: Repaired mesh
        warnings: List of warnings
    """
    warnings = []

    if len(mesh.triangles) == 0:
        warnings.append("Empty mesh - cannot repair")
        return mesh, warnings

    # Remove degenerate triangles
    mesh.remove_degenerate_triangles()

    # Remove duplicated triangles
    mesh.remove_duplicated_triangles()

    # Remove duplicated vertices
    mesh.remove_duplicated_vertices()

    # Remove non-manifold edges
    mesh.remove_non_manifold_edges()

    # Check if watertight
    if not mesh.is_watertight():
        warnings.append("Mesh has holes - volume calculation may be less accurate")

        # Try to fill small holes
        # Note: Open3D doesn't have built-in hole filling
        # For production, consider using pymeshlab or trimesh
        pass

    # Compute vertex normals
    mesh.compute_vertex_normals()

    return mesh, warnings


def smooth_mesh(
    mesh: o3d.geometry.TriangleMesh,
    iterations: int = 5,
) -> o3d.geometry.TriangleMesh:
    """
    Smooth mesh using Laplacian smoothing.

    Args:
        mesh: Input mesh
        iterations: Number of smoothing iterations

    Returns:
        smoothed mesh
    """
    mesh = mesh.filter_smooth_laplacian(
        number_of_iterations=iterations,
        lambda_filter=0.5,
    )
    mesh.compute_vertex_normals()
    return mesh


async def generate_mesh(
    point_cloud: o3d.geometry.PointCloud,
) -> tuple[o3d.geometry.TriangleMesh, list[str]]:
    """
    Generate mesh from point cloud.

    Tries Poisson reconstruction first, falls back to Ball Pivoting.

    Args:
        point_cloud: Dense point cloud

    Returns:
        mesh: Triangle mesh
        warnings: List of warnings
    """
    loop = asyncio.get_event_loop()
    all_warnings = []

    # Try Poisson reconstruction first
    mesh, warnings = await loop.run_in_executor(
        None,
        generate_poisson_mesh,
        point_cloud,
    )
    all_warnings.extend(warnings)

    # If Poisson failed, try Ball Pivoting
    if len(mesh.triangles) == 0:
        all_warnings.append("Poisson reconstruction failed, trying Ball Pivoting")
        mesh, warnings = await loop.run_in_executor(
            None,
            generate_ball_pivoting_mesh,
            point_cloud,
        )
        all_warnings.extend(warnings)

    # Repair mesh
    mesh, warnings = await loop.run_in_executor(
        None,
        repair_mesh,
        mesh,
    )
    all_warnings.extend(warnings)

    # Light smoothing
    if len(mesh.triangles) > 0:
        mesh = await loop.run_in_executor(
            None,
            smooth_mesh,
            mesh,
            3,
        )

    return mesh, all_warnings
