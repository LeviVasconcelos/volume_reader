import numpy as np
import pytest
import open3d as o3d

from app.pipeline.volume import (
    compute_adaptive_voxel_size,
    voxelize_mesh,
    compute_volume_from_voxels,
    extract_dimensions,
    compute_scale_from_user_dimensions,
    calculate_volume,
)
from app.models import UserDimensionsScaleData


def create_test_box_mesh(size_x: float, size_y: float, size_z: float) -> o3d.geometry.TriangleMesh:
    """Create a simple box mesh for testing."""
    mesh = o3d.geometry.TriangleMesh.create_box(size_x, size_y, size_z)
    mesh.compute_vertex_normals()
    return mesh


def create_test_sphere_mesh(radius: float) -> o3d.geometry.TriangleMesh:
    """Create a sphere mesh for testing."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius)
    mesh.compute_vertex_normals()
    return mesh


class TestAdaptiveVoxelSize:
    def test_small_mesh(self):
        mesh = create_test_box_mesh(1.0, 0.5, 0.1)
        voxel_size = compute_adaptive_voxel_size(mesh)

        assert voxel_size > 0
        # Should be small enough for detail but not too small
        assert 0.001 < voxel_size < 0.1

    def test_large_mesh(self):
        mesh = create_test_box_mesh(100.0, 50.0, 10.0)
        voxel_size = compute_adaptive_voxel_size(mesh)

        assert voxel_size > 0
        # Larger mesh should have larger voxel size
        assert voxel_size > 0.05


class TestVoxelization:
    def test_voxelize_box(self):
        mesh = create_test_box_mesh(1.0, 1.0, 1.0)
        voxel_grid = voxelize_mesh(mesh, voxel_size=0.1)

        voxels = voxel_grid.get_voxels()
        # 1x1x1 box with 0.1 voxel size should have ~1000 voxels
        assert 500 < len(voxels) < 1500


class TestVolumeCalculation:
    def test_unit_cube_volume(self):
        mesh = create_test_box_mesh(1.0, 1.0, 1.0)
        voxel_grid = voxelize_mesh(mesh, voxel_size=0.05)

        # Scale factor of 1000 means mesh units = meters, output in liters
        # 1m³ = 1000 liters
        volume = compute_volume_from_voxels(voxel_grid, scale_factor=1000.0)

        # Should be close to 1000 liters (1m³)
        assert 900 < volume < 1100

    def test_scaled_box_volume(self):
        # Box of 2 x 0.5 x 0.1 mesh units
        mesh = create_test_box_mesh(2.0, 0.5, 0.1)
        voxel_grid = voxelize_mesh(mesh, voxel_size=0.02)

        # If mesh units are already in mm (scale_factor=1), volume in mm³
        volume = compute_volume_from_voxels(voxel_grid, scale_factor=1.0)

        # 2 * 0.5 * 0.1 = 0.1 mm³ = 0.0000001 liters
        # With voxelization error, should be close
        assert volume < 0.001  # Very small in liters


class TestDimensionExtraction:
    def test_extract_box_dimensions(self):
        # Create box: 2m x 0.5m x 0.1m (surfboard-like)
        mesh = create_test_box_mesh(2.0, 0.5, 0.1)

        # Scale factor: mesh units to mm (let's say mesh is in meters)
        scale_factor = 1000.0  # 1 mesh unit = 1000mm = 1m

        length, width, thickness = extract_dimensions(mesh, scale_factor)

        # Dimensions should be sorted: length > width > thickness
        assert length > width > thickness
        assert 1900 < length < 2100  # ~2000mm
        assert 450 < width < 550     # ~500mm
        assert 90 < thickness < 110   # ~100mm


class TestScaleFromUserDimensions:
    def test_scale_from_length_only(self):
        mesh = create_test_box_mesh(2.0, 0.5, 0.1)
        user_dims = UserDimensionsScaleData(length_mm=1800)

        scale_factor, warnings = compute_scale_from_user_dimensions(mesh, user_dims)

        # Mesh length is 2.0 units, user says 1800mm
        # Scale factor should be ~900 (1800/2.0)
        assert 850 < scale_factor < 950

    def test_scale_from_multiple_dimensions(self):
        mesh = create_test_box_mesh(2.0, 0.5, 0.1)
        user_dims = UserDimensionsScaleData(
            length_mm=2000,
            width_mm=500,
            thickness_mm=100
        )

        scale_factor, warnings = compute_scale_from_user_dimensions(mesh, user_dims)

        # All dimensions suggest scale of 1000
        assert 950 < scale_factor < 1050
        assert len(warnings) == 0

    def test_inconsistent_dimensions_warning(self):
        mesh = create_test_box_mesh(2.0, 0.5, 0.1)
        user_dims = UserDimensionsScaleData(
            length_mm=2000,  # suggests scale 1000
            width_mm=250,    # suggests scale 500 (inconsistent!)
        )

        scale_factor, warnings = compute_scale_from_user_dimensions(mesh, user_dims)

        # Should warn about inconsistency
        assert any("inconsistent" in w.lower() for w in warnings)


@pytest.mark.asyncio
async def test_calculate_volume_surfboard_like():
    # Create a surfboard-like shape (elongated box)
    # Mesh in "meters": 1.8m x 0.5m x 0.06m
    mesh = create_test_box_mesh(1.8, 0.5, 0.06)

    # User provides dimensions in mm
    user_dims = UserDimensionsScaleData(
        length_mm=1800,
        width_mm=500,
        thickness_mm=60
    )

    volume, dimensions, warnings = await calculate_volume(
        mesh,
        scale_factor=1.0,  # Will be computed from user dims
        user_dimensions=user_dims,
    )

    # Expected volume: 1800 * 500 * 60 = 54,000,000 mm³ = 54 liters
    # Box volume, actual surfboard would be less due to shape
    assert 40 < volume < 70  # Allow for voxelization error

    # Check dimensions
    assert 1700 < dimensions[0] < 1900  # length
    assert 450 < dimensions[1] < 550    # width
    assert 50 < dimensions[2] < 70      # thickness


@pytest.mark.asyncio
async def test_calculate_volume_empty_mesh():
    mesh = o3d.geometry.TriangleMesh()

    volume, dimensions, warnings = await calculate_volume(
        mesh,
        scale_factor=1.0,
        user_dimensions=None,
    )

    assert volume == 0.0
    assert any("empty mesh" in w.lower() for w in warnings)
