#!/usr/bin/env python3
"""
CLI runner for debug pipeline.

Usage:
    python run_debug_pipeline.py ./images/ ./debug_output/ \\
        --scale-method user_dimensions \\
        --length 1830 --width 520 --thickness 63 \\
        --click-image 0 --click-x 0.5 --click-y 0.5
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


async def main():
    parser = argparse.ArgumentParser(
        description="Run reconstruction pipeline with debug output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # User dimensions mode (most common):
    python run_debug_pipeline.py ./images/ ./debug_output/ \\
        --length 1830 --width 520 --thickness 63

    # ArUco marker mode:
    python run_debug_pipeline.py ./images/ ./debug_output/ \\
        --scale-method aruco --aruco-id 0 --aruco-size 50

    # With custom click point:
    python run_debug_pipeline.py ./images/ ./debug_output/ \\
        --length 1830 --click-image 2 --click-x 0.4 --click-y 0.6
        """
    )

    parser.add_argument("image_dir", type=Path, help="Directory containing input images")
    parser.add_argument("output_dir", type=Path, help="Directory for debug output")

    # Scale options
    parser.add_argument("--scale-method", choices=["user_dimensions", "aruco", "credit_card"],
                       default="user_dimensions", help="Scale detection method")
    parser.add_argument("--length", type=float, help="Board length in mm")
    parser.add_argument("--width", type=float, help="Board width in mm")
    parser.add_argument("--thickness", type=float, help="Board thickness in mm")
    parser.add_argument("--aruco-id", type=int, default=0, help="ArUco marker ID")
    parser.add_argument("--aruco-size", type=float, default=50.0, help="ArUco marker size in mm")

    # Click point options
    parser.add_argument("--click-image", type=int, default=0, help="Image index for click point")
    parser.add_argument("--click-x", type=float, default=0.5, help="Normalized X coordinate (0-1)")
    parser.add_argument("--click-y", type=float, default=0.5, help="Normalized Y coordinate (0-1)")

    # Processing options
    parser.add_argument("--max-images", type=int, help="Maximum number of images to process")
    parser.add_argument("--skip-stages", nargs="+", default=[],
                       help="Stages to skip (e.g., dense mesh)")

    args = parser.parse_args()

    # Validate inputs
    if not args.image_dir.exists():
        print(f"Error: Image directory not found: {args.image_dir}")
        sys.exit(1)

    # Find images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths = sorted([
        p for p in args.image_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if not image_paths:
        print(f"Error: No images found in {args.image_dir}")
        sys.exit(1)

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Found {len(image_paths)} images")

    # Import pipeline components
    from app.pipeline.preprocess import preprocess_images, CameraIntrinsics
    from app.pipeline.segmentation import segment_board
    from app.pipeline.scale import detect_scale
    from app.pipeline.features import extract_and_match_features
    from app.pipeline.sfm import run_sfm, CameraPose
    from app.pipeline.dense import densify_point_cloud
    from app.pipeline.mesh import generate_mesh
    from app.pipeline.volume import calculate_volume, voxelize_mesh, compute_adaptive_voxel_size
    from app.pipeline.confidence import calculate_confidence
    from app.models import (
        ClickPoint, ScaleMethod, ArucoScaleData,
        CreditCardScaleData, UserDimensionsScaleData
    )
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

    # Create debug context
    debug = DebugContext(args.output_dir, enabled=True)
    print(f"Debug output directory: {args.output_dir}")

    warnings = []

    try:
        # Stage 1: Preprocessing
        print("\n=== Stage 1: Preprocessing ===")
        images, intrinsics = await preprocess_images(image_paths)
        print(f"Preprocessed {len(images)} images ({images[0].shape[1]}x{images[0].shape[0]})")
        debug_preprocessing(debug, images, intrinsics, image_paths)
        print(f"  -> Debug output: {debug.stage_dir('preprocess')}")

        # Stage 2: Segmentation
        print("\n=== Stage 2: Segmentation ===")
        click_point = ClickPoint(
            image_index=args.click_image,
            x=args.click_x,
            y=args.click_y
        )
        masks = await segment_board(images, click_point)
        print(f"Generated {len(masks)} segmentation masks")
        debug_segmentation(debug, images, masks, (click_point.x, click_point.y), click_point.image_index)
        print(f"  -> Debug output: {debug.stage_dir('segmentation')}")

        # Stage 3: Scale Detection
        print("\n=== Stage 3: Scale Detection ===")
        scale_method = ScaleMethod(args.scale_method)

        aruco_data = None
        credit_card_data = None
        user_dimensions = None

        if scale_method == ScaleMethod.ARUCO:
            aruco_data = ArucoScaleData(aruco_id=args.aruco_id, aruco_size_mm=args.aruco_size)
        elif scale_method == ScaleMethod.CREDIT_CARD:
            credit_card_data = CreditCardScaleData()
        else:  # user_dimensions
            user_dimensions = UserDimensionsScaleData(
                length_mm=args.length,
                width_mm=args.width,
                thickness_mm=args.thickness
            )

        scale_factor, scale_warnings = await detect_scale(
            images, masks, scale_method, aruco_data, credit_card_data, user_dimensions
        )
        warnings.extend(scale_warnings)
        print(f"Scale factor: {scale_factor:.6f} (method: {scale_method.value})")

        # Build detections list for debug (simplified)
        scale_detections = [{"scale": scale_factor}] * len(images)
        debug_scale_detection(debug, images, scale_factor, scale_method.value, scale_detections)
        print(f"  -> Debug output: {debug.stage_dir('scale')}")

        # Stage 4: Feature Extraction
        print("\n=== Stage 4: Feature Extraction ===")
        keypoints, descriptors, matches = await extract_and_match_features(images, masks)
        total_matches = sum(len(m) for m in matches.values())
        print(f"Extracted features: {sum(len(kp) for kp in keypoints)} keypoints, {total_matches} matches")
        debug_features(debug, images, masks, keypoints, matches)
        print(f"  -> Debug output: {debug.stage_dir('features')}")

        # Stage 5: Structure from Motion
        print("\n=== Stage 5: SfM ===")
        sparse_cloud, camera_poses = await run_sfm(images, keypoints, matches, intrinsics)
        print(f"Sparse reconstruction: {len(sparse_cloud.points)} points, {len(camera_poses)} cameras")
        debug_sfm(debug, sparse_cloud, camera_poses)
        print(f"  -> Debug output: {debug.stage_dir('sfm')}")

        # Stage 6: Dense Reconstruction
        print("\n=== Stage 6: Dense Reconstruction ===")
        if "dense" not in args.skip_stages:
            dense_cloud, dense_warnings = await densify_point_cloud(
                images, masks, camera_poses, intrinsics
            )
            warnings.extend(dense_warnings)
            print(f"Dense cloud: {len(dense_cloud.points)} points")

            # Collect depth maps for debug (simplified - would need to store during dense)
            depth_maps = []  # Would be populated during actual dense reconstruction
            debug_dense(debug, depth_maps, dense_cloud)
            print(f"  -> Debug output: {debug.stage_dir('dense')}")
        else:
            print("  Skipped")
            import open3d as o3d
            dense_cloud = o3d.geometry.PointCloud()

        # Stage 7: Mesh Generation
        print("\n=== Stage 7: Mesh Generation ===")
        if "mesh" not in args.skip_stages:
            mesh, mesh_warnings = await generate_mesh(dense_cloud if len(dense_cloud.points) > 0 else sparse_cloud)
            warnings.extend(mesh_warnings)
            print(f"Generated mesh: {len(mesh.triangles)} triangles")
            debug_mesh(debug, mesh)
            print(f"  -> Debug output: {debug.stage_dir('mesh')}")
        else:
            print("  Skipped")
            import open3d as o3d
            mesh = o3d.geometry.TriangleMesh()

        # Stage 8: Volume Calculation
        print("\n=== Stage 8: Volume Calculation ===")
        if len(mesh.triangles) > 0:
            volume_liters, dimensions, volume_warnings = await calculate_volume(
                mesh, scale_factor, user_dimensions
            )
            warnings.extend(volume_warnings)
            print(f"Volume: {volume_liters:.2f} liters")
            print(f"Dimensions: {dimensions[0]:.0f} x {dimensions[1]:.0f} x {dimensions[2]:.0f} mm")

            # Create voxel grid for debug
            voxel_size = compute_adaptive_voxel_size(mesh)
            voxel_grid = voxelize_mesh(mesh, voxel_size)
            debug_volume(debug, mesh, voxel_grid, voxel_size, dimensions, scale_factor, volume_liters)
            print(f"  -> Debug output: {debug.stage_dir('volume')}")
        else:
            print("  Skipped (no mesh)")
            volume_liters = 0.0
            dimensions = (0.0, 0.0, 0.0)

        # Stage 9: Confidence Scoring
        print("\n=== Stage 9: Confidence Scoring ===")
        confidence = await calculate_confidence(
            num_images=len(images),
            num_matches=total_matches,
            sparse_points=len(sparse_cloud.points),
            dense_points=len(dense_cloud.points),
            mesh_quality=mesh,
            scale_factor=scale_factor,
            warnings=warnings,
        )
        print(f"Confidence: {confidence:.2f}")

        # Build component scores for debug
        component_scores = {
            "images": min(len(images) / 30, 1.0),
            "matches": min(total_matches / len(images) / 500, 1.0) if len(images) > 0 else 0,
            "dense_points": min(len(dense_cloud.points) / 100000, 1.0),
            "mesh_quality": 1.0 if mesh.is_watertight() else 0.7 if len(mesh.triangles) > 5000 else 0.4,
            "scale": 1.0 if scale_factor != 1.0 else 0.6,
            "warnings": max(0, 1.0 - len(warnings) * 0.15),
        }
        weights = {
            "images": 0.15,
            "matches": 0.15,
            "dense_points": 0.2,
            "mesh_quality": 0.25,
            "scale": 0.15,
            "warnings": 0.1,
        }
        debug_confidence(debug, component_scores, weights, confidence, warnings)
        print(f"  -> Debug output: {debug.stage_dir('confidence')}")

        # Stage 10: Object Verification
        print("\n=== Stage 10: Object Verification ===")
        verification_results = debug_object_verification(
            debug, images, masks, sparse_cloud, dense_cloud, camera_poses, intrinsics
        )
        if verification_results:
            print(f"Mean IoU: {verification_results['aggregate']['mean_iou']:.3f}")
            low_iou = verification_results['aggregate']['low_iou_images']
            if low_iou:
                print(f"  Warning: Low IoU images: {low_iou}")
        print(f"  -> Debug output: {debug.stage_dir('object_verification')}")

        # Summary
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETE")
        print("=" * 50)
        print(f"Volume: {volume_liters:.2f} liters")
        print(f"Dimensions: {dimensions[0]:.0f} x {dimensions[1]:.0f} x {dimensions[2]:.0f} mm")
        print(f"Confidence: {confidence:.2f}")
        print(f"Warnings: {len(warnings)}")
        for w in warnings:
            print(f"  - {w}")
        print(f"\nDebug output saved to: {args.output_dir}")
        print("\nTo view results:")
        print(f"  - Open PLY files in MeshLab or CloudCompare")
        print(f"  - View PNG visualizations in any image viewer")
        print(f"  - Check JSON files for detailed metrics")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
