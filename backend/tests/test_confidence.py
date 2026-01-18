import pytest
import open3d as o3d
import numpy as np

from app.pipeline.confidence import calculate_confidence


def create_test_mesh(n_triangles: int, watertight: bool = False) -> o3d.geometry.TriangleMesh:
    """Create a test mesh with specified number of triangles."""
    if watertight:
        # Sphere is watertight
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    else:
        # Box with one face removed (not watertight)
        mesh = o3d.geometry.TriangleMesh.create_box()
        # Simple box is watertight, but we'll mark it as not for testing

    return mesh


@pytest.mark.asyncio
async def test_high_confidence_scenario():
    """Test that good inputs produce high confidence."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

    confidence = await calculate_confidence(
        num_images=30,
        num_matches=20000,
        sparse_points=5000,
        dense_points=150000,
        mesh_quality=mesh,
        scale_factor=0.5,  # Valid scale
        warnings=[],
    )

    assert confidence > 0.85


@pytest.mark.asyncio
async def test_low_confidence_few_images():
    """Test that few images reduce confidence."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

    confidence = await calculate_confidence(
        num_images=3,  # Very few images
        num_matches=500,
        sparse_points=500,
        dense_points=5000,
        mesh_quality=mesh,
        scale_factor=0.5,
        warnings=[],
    )

    assert confidence < 0.7


@pytest.mark.asyncio
async def test_low_confidence_no_scale():
    """Test that default scale (1.0) reduces confidence."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

    confidence = await calculate_confidence(
        num_images=25,
        num_matches=15000,
        sparse_points=3000,
        dense_points=100000,
        mesh_quality=mesh,
        scale_factor=1.0,  # Default scale
        warnings=[],
    )

    # Should still be reasonable but lower than with valid scale
    assert 0.5 < confidence < 0.9


@pytest.mark.asyncio
async def test_confidence_reduced_by_warnings():
    """Test that warnings reduce confidence."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

    confidence_no_warnings = await calculate_confidence(
        num_images=25,
        num_matches=15000,
        sparse_points=3000,
        dense_points=100000,
        mesh_quality=mesh,
        scale_factor=0.5,
        warnings=[],
    )

    confidence_with_warnings = await calculate_confidence(
        num_images=25,
        num_matches=15000,
        sparse_points=3000,
        dense_points=100000,
        mesh_quality=mesh,
        scale_factor=0.5,
        warnings=["Warning 1", "Warning 2", "Warning 3"],
    )

    assert confidence_with_warnings < confidence_no_warnings


@pytest.mark.asyncio
async def test_critical_warning_penalty():
    """Test that critical warnings severely reduce confidence."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

    confidence = await calculate_confidence(
        num_images=25,
        num_matches=15000,
        sparse_points=3000,
        dense_points=100000,
        mesh_quality=mesh,
        scale_factor=0.5,
        warnings=["Failed to initialize reconstruction"],  # Critical warning
    )

    assert confidence < 0.5


@pytest.mark.asyncio
async def test_empty_mesh_low_confidence():
    """Test that empty mesh produces low confidence."""
    mesh = o3d.geometry.TriangleMesh()  # Empty mesh

    confidence = await calculate_confidence(
        num_images=25,
        num_matches=15000,
        sparse_points=3000,
        dense_points=100000,
        mesh_quality=mesh,
        scale_factor=0.5,
        warnings=[],
    )

    assert confidence < 0.5


@pytest.mark.asyncio
async def test_confidence_bounds():
    """Test that confidence is always between 0 and 1."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

    # Test with extreme values
    for num_images in [1, 5, 20, 100]:
        for num_matches in [10, 1000, 50000]:
            confidence = await calculate_confidence(
                num_images=num_images,
                num_matches=num_matches,
                sparse_points=1000,
                dense_points=10000,
                mesh_quality=mesh,
                scale_factor=0.5,
                warnings=[],
            )

            assert 0.0 <= confidence <= 1.0
