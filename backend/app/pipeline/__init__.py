import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from app.models import JobResult, Dimensions

if TYPE_CHECKING:
    from app.jobs import Job

from app.pipeline.preprocess import preprocess_images
from app.pipeline.segmentation import segment_board
from app.pipeline.scale import detect_scale
from app.pipeline.features import extract_and_match_features
from app.pipeline.sfm import run_sfm
from app.pipeline.dense import densify_point_cloud
from app.pipeline.mesh import generate_mesh
from app.pipeline.volume import calculate_volume, voxelize_mesh, compute_adaptive_voxel_size
from app.pipeline.confidence import calculate_confidence


async def run_pipeline(job: "Job", debug_dir: Optional[Path] = None):
    """
    Main pipeline orchestrator.
    Runs all reconstruction stages and updates job status.

    Args:
        job: Job instance containing request parameters and image paths
        debug_dir: Optional directory to save debug visualizations
    """
    warnings: list[str] = []

    # Initialize debug context if debug_dir provided
    debug = None
    if debug_dir:
        from app.pipeline.debug import (
            DebugContext,
            debug_preprocessing,
            debug_segmentation,
            debug_scale_detection,
            debug_features,
            debug_sfm,
            debug_dense,
            debug_mesh,
            debug_volume,
            debug_confidence,
            debug_object_verification,
        )
        debug = DebugContext(debug_dir, enabled=True)

    try:
        job.set_processing("Preprocessing images")

        # Stage 1: Preprocess images
        images, camera_intrinsics = await preprocess_images(job.image_paths)
        job.set_progress("Images preprocessed")
        if debug:
            debug_preprocessing(debug, images, camera_intrinsics, job.image_paths)

        # Stage 2: Segment the surfboard
        job.set_progress("Segmenting surfboard")
        masks = await segment_board(
            images,
            job.request.board_click_point,
        )
        job.set_progress("Segmentation complete")
        if debug:
            click_pt = job.request.board_click_point
            debug_segmentation(debug, images, masks, (click_pt.x, click_pt.y), click_pt.image_index)

        # Stage 3: Detect scale reference
        job.set_progress("Detecting scale reference")
        scale_factor, scale_warnings = await detect_scale(
            images,
            masks,
            job.request.scale_method,
            job.request.aruco_data,
            job.request.credit_card_data,
            job.request.user_dimensions,
        )
        warnings.extend(scale_warnings)
        job.set_progress("Scale reference detected")
        if debug:
            scale_detections = [{"scale": scale_factor}] * len(images)
            debug_scale_detection(debug, images, scale_factor, job.request.scale_method.value, scale_detections)

        # Stage 4: Extract and match features
        job.set_progress("Extracting features")
        keypoints, descriptors, matches = await extract_and_match_features(
            images, masks
        )
        job.set_progress("Features matched")
        if debug:
            debug_features(debug, images, masks, keypoints, matches)

        # Stage 5: Structure from Motion
        job.set_progress("Running Structure from Motion")
        sparse_cloud, camera_poses = await run_sfm(
            images, keypoints, matches, camera_intrinsics
        )
        job.set_progress("Sparse reconstruction complete")
        if debug:
            debug_sfm(debug, sparse_cloud, camera_poses)

        # Stage 6: Dense reconstruction
        job.set_progress("Densifying point cloud")
        dense_cloud, dense_warnings = await densify_point_cloud(
            images, masks, camera_poses, camera_intrinsics
        )
        warnings.extend(dense_warnings)
        job.set_progress("Dense reconstruction complete")
        if debug:
            debug_dense(debug, [], dense_cloud)

        # Stage 7: Generate mesh
        job.set_progress("Generating mesh")
        mesh, mesh_warnings = await generate_mesh(dense_cloud)
        warnings.extend(mesh_warnings)
        job.set_progress("Mesh generated")
        if debug:
            debug_mesh(debug, mesh)

        # Stage 8: Calculate volume with scale
        job.set_progress("Calculating volume")
        volume_liters, dimensions, volume_warnings = await calculate_volume(
            mesh, scale_factor, job.request.user_dimensions
        )
        warnings.extend(volume_warnings)
        job.set_progress("Volume calculated")
        if debug and len(mesh.triangles) > 0:
            voxel_size = compute_adaptive_voxel_size(mesh)
            voxel_grid = voxelize_mesh(mesh, voxel_size)
            debug_volume(debug, mesh, voxel_grid, voxel_size, dimensions, scale_factor, volume_liters)

        # Stage 9: Calculate confidence
        confidence = await calculate_confidence(
            num_images=len(images),
            num_matches=sum(len(m) for m in matches.values()),
            sparse_points=len(sparse_cloud.points) if hasattr(sparse_cloud, 'points') else 0,
            dense_points=len(dense_cloud.points) if hasattr(dense_cloud, 'points') else 0,
            mesh_quality=mesh,
            scale_factor=scale_factor,
            warnings=warnings,
        )
        if debug:
            total_matches = sum(len(m) for m in matches.values())
            component_scores = {
                "images": min(len(images) / 30, 1.0),
                "matches": min(total_matches / len(images) / 500, 1.0) if len(images) > 0 else 0,
                "dense_points": min(len(dense_cloud.points) / 100000, 1.0) if hasattr(dense_cloud, 'points') else 0,
                "mesh_quality": 1.0 if mesh.is_watertight() else (0.7 if len(mesh.triangles) > 5000 else 0.4),
                "scale": 1.0 if scale_factor != 1.0 else 0.6,
                "warnings": max(0, 1.0 - len(warnings) * 0.15),
            }
            weights = {"images": 0.15, "matches": 0.15, "dense_points": 0.2, "mesh_quality": 0.25, "scale": 0.15, "warnings": 0.1}
            debug_confidence(debug, component_scores, weights, confidence, warnings)

        # Stage 10: Object verification (debug only)
        if debug:
            debug_object_verification(debug, images, masks, sparse_cloud, dense_cloud, camera_poses, camera_intrinsics)

        # Assemble result
        result = JobResult(
            volume_liters=volume_liters,
            dimensions=Dimensions(
                length_mm=dimensions[0],
                width_mm=dimensions[1],
                thickness_mm=dimensions[2],
            ),
            confidence=confidence,
            warnings=warnings,
        )

        if confidence < 0.6:
            job.set_partial(result, warnings)
        else:
            job.set_completed(result)

    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        traceback.print_exc()
        job.set_failed([error_msg])
