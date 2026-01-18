import asyncio
from typing import Optional

import open3d as o3d


async def calculate_confidence(
    num_images: int,
    num_matches: int,
    sparse_points: int,
    dense_points: int,
    mesh_quality: o3d.geometry.TriangleMesh,
    scale_factor: float,
    warnings: list[str],
) -> float:
    """
    Calculate confidence score for the volume estimation.

    Factors considered:
    - Number of input images
    - Feature match count
    - Point cloud density
    - Mesh quality (watertight, triangle count)
    - Scale factor validity
    - Number of warnings generated

    Args:
        num_images: Number of input images
        num_matches: Total number of feature matches
        sparse_points: Number of points in sparse reconstruction
        dense_points: Number of points in dense reconstruction
        mesh_quality: Generated mesh
        scale_factor: Scale factor used
        warnings: Warnings generated during processing

    Returns:
        confidence: Score between 0.0 and 1.0
    """
    scores = []
    weights = []

    # 1. Image count score (weight: 0.15)
    # More images = better coverage
    # Ideal: 20-40 images
    if num_images >= 30:
        img_score = 1.0
    elif num_images >= 20:
        img_score = 0.9
    elif num_images >= 10:
        img_score = 0.7
    elif num_images >= 5:
        img_score = 0.5
    else:
        img_score = 0.3
    scores.append(img_score)
    weights.append(0.15)

    # 2. Feature match score (weight: 0.15)
    # More matches = better reconstruction
    matches_per_image = num_matches / max(num_images, 1)
    if matches_per_image >= 500:
        match_score = 1.0
    elif matches_per_image >= 200:
        match_score = 0.8
    elif matches_per_image >= 100:
        match_score = 0.6
    elif matches_per_image >= 50:
        match_score = 0.4
    else:
        match_score = 0.2
    scores.append(match_score)
    weights.append(0.15)

    # 3. Dense point cloud score (weight: 0.2)
    # Higher density = more accurate surface
    if dense_points >= 100000:
        dense_score = 1.0
    elif dense_points >= 50000:
        dense_score = 0.85
    elif dense_points >= 20000:
        dense_score = 0.7
    elif dense_points >= 5000:
        dense_score = 0.5
    else:
        dense_score = 0.3
    scores.append(dense_score)
    weights.append(0.2)

    # 4. Mesh quality score (weight: 0.25)
    mesh_score = 0.5  # Default
    if len(mesh_quality.triangles) > 0:
        # Check if watertight
        if mesh_quality.is_watertight():
            mesh_score = 1.0
        else:
            # Score based on triangle count
            n_triangles = len(mesh_quality.triangles)
            if n_triangles >= 50000:
                mesh_score = 0.8
            elif n_triangles >= 20000:
                mesh_score = 0.7
            elif n_triangles >= 5000:
                mesh_score = 0.5
            else:
                mesh_score = 0.3
    else:
        mesh_score = 0.0
    scores.append(mesh_score)
    weights.append(0.25)

    # 5. Scale factor score (weight: 0.15)
    # Valid scale factor is critical for accuracy
    if scale_factor > 0 and scale_factor != 1.0:
        scale_score = 1.0
    elif scale_factor == 1.0:
        # Using default scale - less confident
        scale_score = 0.6
    else:
        scale_score = 0.0
    scores.append(scale_score)
    weights.append(0.15)

    # 6. Warning penalty (weight: 0.1)
    # Each warning reduces confidence
    warning_score = max(0.0, 1.0 - len(warnings) * 0.15)
    scores.append(warning_score)
    weights.append(0.1)

    # Compute weighted average
    confidence = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    # Apply hard penalties for critical issues
    critical_warnings = [
        "Empty mesh",
        "Failed to initialize",
        "Scale reference not detected",
    ]
    for warning in warnings:
        for critical in critical_warnings:
            if critical.lower() in warning.lower():
                confidence *= 0.5
                break

    # Clamp to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    return confidence
