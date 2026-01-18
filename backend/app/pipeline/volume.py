import asyncio
from typing import Optional

import numpy as np
import open3d as o3d

from app.models import UserDimensionsScaleData


def compute_adaptive_voxel_size(
    mesh: o3d.geometry.TriangleMesh,
    target_resolution_mm: float = 1.5,
) -> float:
    """
    Compute adaptive voxel size based on mesh dimensions.

    Targets ~1-2mm resolution for accurate volume calculation.

    Args:
        mesh: Input mesh
        target_resolution_mm: Target resolution in mm

    Returns:
        voxel_size: Voxel size to use
    """
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    max_dim = max(extent)

    # Estimate mesh scale (assuming surfboard ~2m max)
    # This will be refined with actual scale factor
    estimated_scale = 2000 / max_dim if max_dim > 0 else 1.0

    # Voxel size in mesh units
    voxel_size = target_resolution_mm / estimated_scale

    # Clamp to reasonable range
    voxel_size = max(voxel_size, max_dim / 1000)  # At least 1000 voxels along max dim
    voxel_size = min(voxel_size, max_dim / 100)   # At most 100 voxels along max dim

    return voxel_size


def voxelize_mesh(
    mesh: o3d.geometry.TriangleMesh,
    voxel_size: float,
) -> o3d.geometry.VoxelGrid:
    """
    Convert mesh to voxel grid.

    Args:
        mesh: Input triangle mesh
        voxel_size: Size of each voxel

    Returns:
        voxel_grid: Voxelized representation
    """
    # Create voxel grid from mesh
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh,
        voxel_size=voxel_size,
    )

    return voxel_grid


def compute_volume_from_voxels(
    voxel_grid: o3d.geometry.VoxelGrid,
    scale_factor: float,
) -> float:
    """
    Compute volume from voxel grid.

    Args:
        voxel_grid: Voxelized mesh
        scale_factor: Conversion from mesh units to mm

    Returns:
        volume_liters: Volume in liters
    """
    # Get number of voxels
    voxels = voxel_grid.get_voxels()
    n_voxels = len(voxels)

    # Get voxel size in mesh units
    voxel_size = voxel_grid.voxel_size

    # Volume in mesh units cubed
    volume_mesh_units = n_voxels * (voxel_size ** 3)

    # Convert to mm³ using scale factor
    # scale_factor is mm per mesh unit
    volume_mm3 = volume_mesh_units * (scale_factor ** 3)

    # Convert to liters (1 liter = 1,000,000 mm³)
    volume_liters = volume_mm3 / 1_000_000

    return volume_liters


def extract_dimensions(
    mesh: o3d.geometry.TriangleMesh,
    scale_factor: float,
) -> tuple[float, float, float]:
    """
    Extract board dimensions (length, width, thickness) from mesh.

    Uses oriented bounding box to find principal dimensions.

    Args:
        mesh: Input mesh
        scale_factor: Conversion from mesh units to mm

    Returns:
        (length_mm, width_mm, thickness_mm)
    """
    # Get oriented bounding box
    obb = mesh.get_oriented_bounding_box()
    extent = obb.extent

    # Sort dimensions (largest to smallest)
    dims = sorted(extent, reverse=True)

    # Apply scale factor
    length_mm = dims[0] * scale_factor
    width_mm = dims[1] * scale_factor
    thickness_mm = dims[2] * scale_factor

    return length_mm, width_mm, thickness_mm


def compute_scale_from_user_dimensions(
    mesh: o3d.geometry.TriangleMesh,
    user_dimensions: UserDimensionsScaleData,
) -> tuple[float, list[str]]:
    """
    Compute scale factor by fitting mesh to user-provided dimensions.

    Args:
        mesh: Input mesh
        user_dimensions: User-provided dimensions

    Returns:
        scale_factor: Computed scale factor (mm per mesh unit)
        warnings: List of warnings
    """
    warnings = []

    # Get mesh dimensions
    obb = mesh.get_oriented_bounding_box()
    mesh_dims = sorted(obb.extent, reverse=True)  # length, width, thickness

    scale_factors = []

    # Match provided dimensions to mesh dimensions
    if user_dimensions.length_mm is not None:
        scale_factors.append(user_dimensions.length_mm / mesh_dims[0])

    if user_dimensions.width_mm is not None:
        scale_factors.append(user_dimensions.width_mm / mesh_dims[1])

    if user_dimensions.thickness_mm is not None:
        scale_factors.append(user_dimensions.thickness_mm / mesh_dims[2])

    if not scale_factors:
        warnings.append("No dimensions provided for scaling")
        return 1.0, warnings

    # Use median scale factor
    scale_factor = float(np.median(scale_factors))

    # Check consistency
    if len(scale_factors) >= 2:
        scale_std = np.std(scale_factors)
        scale_mean = np.mean(scale_factors)
        if scale_std / scale_mean > 0.1:  # More than 10% variation
            warnings.append(
                "Provided dimensions are inconsistent with mesh proportions. "
                "Using average scale factor."
            )

    return scale_factor, warnings


async def calculate_volume(
    mesh: o3d.geometry.TriangleMesh,
    scale_factor: float,
    user_dimensions: Optional[UserDimensionsScaleData],
) -> tuple[float, tuple[float, float, float], list[str]]:
    """
    Calculate volume and dimensions from mesh.

    Args:
        mesh: Input triangle mesh
        scale_factor: Initial scale factor from marker detection (or 1.0)
        user_dimensions: Optional user-provided dimensions for scaling

    Returns:
        volume_liters: Computed volume in liters
        dimensions: (length_mm, width_mm, thickness_mm)
        warnings: List of warnings
    """
    loop = asyncio.get_event_loop()
    warnings = []

    if len(mesh.triangles) == 0:
        warnings.append("Empty mesh - cannot calculate volume")
        return 0.0, (0.0, 0.0, 0.0), warnings

    # If user provided dimensions, compute scale from them
    if user_dimensions is not None and (
        user_dimensions.length_mm is not None or
        user_dimensions.width_mm is not None or
        user_dimensions.thickness_mm is not None
    ):
        user_scale, scale_warnings = await loop.run_in_executor(
            None,
            compute_scale_from_user_dimensions,
            mesh,
            user_dimensions,
        )
        warnings.extend(scale_warnings)

        # If we also have marker scale, cross-validate
        if scale_factor != 1.0:
            ratio = user_scale / scale_factor
            if abs(ratio - 1.0) > 0.15:  # More than 15% difference
                warnings.append(
                    f"Scale from marker differs from user dimensions by {abs(ratio-1)*100:.1f}%. "
                    "Using user-provided dimensions as primary reference."
                )

        scale_factor = user_scale

    # Compute adaptive voxel size
    voxel_size = await loop.run_in_executor(
        None,
        compute_adaptive_voxel_size,
        mesh,
    )

    # Voxelize mesh
    voxel_grid = await loop.run_in_executor(
        None,
        voxelize_mesh,
        mesh,
        voxel_size,
    )

    # Compute volume
    volume_liters = await loop.run_in_executor(
        None,
        compute_volume_from_voxels,
        voxel_grid,
        scale_factor,
    )

    # Extract dimensions
    dimensions = await loop.run_in_executor(
        None,
        extract_dimensions,
        mesh,
        scale_factor,
    )

    # Sanity check volume for surfboards (typically 20-80 liters)
    if volume_liters < 10:
        warnings.append(
            f"Computed volume ({volume_liters:.1f}L) seems low for a surfboard. "
            "Check scale reference."
        )
    elif volume_liters > 100:
        warnings.append(
            f"Computed volume ({volume_liters:.1f}L) seems high for a surfboard. "
            "Check scale reference."
        )

    return volume_liters, dimensions, warnings
