#!/usr/bin/env python3
"""
Debug script for SAM segmentation with various prompt types.

Usage:
    # Single point (legacy)
    python debug_sam.py ./frames/ ./debug_sam_output/ --click-x 0.5 --click-y 0.5

    # Bounding box
    python debug_sam.py ./frames/ ./debug_sam_output/ --bbox 0.2,0.1,0.8,0.95

    # Multiple landmark points
    python debug_sam.py ./frames/ ./debug_sam_output/ \
        --nose 0.5,0.05 --tail 0.5,0.95 --rail-left 0.3,0.5 --rail-right 0.7,0.5

    # Combined: bbox + points
    python debug_sam.py ./frames/ ./debug_sam_output/ \
        --bbox 0.2,0.1,0.8,0.95 --center 0.5,0.5 --nose 0.5,0.05

    # With refinement
    python debug_sam.py ./frames/ ./debug_sam_output/ --click-x 0.5 --click-y 0.5 --refine 2
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_point(s: str) -> tuple[float, float]:
    """Parse 'x,y' string to tuple."""
    parts = s.split(',')
    return (float(parts[0]), float(parts[1]))


def parse_bbox(s: str) -> tuple[float, float, float, float]:
    """Parse 'x_min,y_min,x_max,y_max' string to tuple."""
    parts = s.split(',')
    return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))


async def main():
    parser = argparse.ArgumentParser(
        description="Debug SAM segmentation with various prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single point
    python debug_sam.py ./frames/ ./out/ --click-x 0.5 --click-y 0.5

    # Bounding box only
    python debug_sam.py ./frames/ ./out/ --bbox 0.2,0.1,0.8,0.95

    # Landmarks (nose, tail, rails)
    python debug_sam.py ./frames/ ./out/ \\
        --nose 0.5,0.05 --tail 0.5,0.95 \\
        --rail-left 0.3,0.5 --rail-right 0.7,0.5

    # Bbox + landmarks combined
    python debug_sam.py ./frames/ ./out/ \\
        --bbox 0.2,0.05,0.8,0.98 \\
        --nose 0.5,0.08 --tail 0.5,0.92 --center 0.5,0.5

    # With iterative refinement
    python debug_sam.py ./frames/ ./out/ --click-x 0.5 --click-y 0.5 --refine 2
        """
    )

    parser.add_argument("image_dir", type=Path, help="Directory containing images")
    parser.add_argument("output_dir", type=Path, help="Debug output directory")

    # Single point (legacy)
    parser.add_argument("--click-image", type=int, default=0, help="Image index for click")
    parser.add_argument("--click-x", type=float, help="Normalized X (0-1)")
    parser.add_argument("--click-y", type=float, help="Normalized Y (0-1)")

    # Bounding box
    parser.add_argument("--bbox", type=str, help="Bounding box: x_min,y_min,x_max,y_max (normalized)")

    # Landmarks
    parser.add_argument("--nose", type=str, help="Nose point: x,y")
    parser.add_argument("--tail", type=str, help="Tail point: x,y")
    parser.add_argument("--rail-left", type=str, help="Left rail point: x,y")
    parser.add_argument("--rail-right", type=str, help="Right rail point: x,y")
    parser.add_argument("--center", type=str, help="Center point: x,y")

    # Extra points
    parser.add_argument("--extra-fg", type=str, action='append', help="Extra foreground point: x,y")
    parser.add_argument("--extra-bg", type=str, action='append', help="Background point (NOT board): x,y")

    # Options
    parser.add_argument("--max-images", type=int, default=5, help="Max images to process")
    parser.add_argument("--refine", type=int, default=0, help="Refinement iterations")

    args = parser.parse_args()

    # Find images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([
        p for p in args.image_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if not image_paths:
        print(f"No images found in {args.image_dir}")
        sys.exit(1)

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Found {len(image_paths)} images")

    # Import modules
    import cv2
    import numpy as np
    from app.pipeline.preprocess import preprocess_images
    from app.pipeline.segmentation import (
        segment_board,
        segment_board_with_refinement,
        segment_with_prompts,
    )
    from app.models import (
        ClickPoint, SegmentationPrompt, BoundingBox, BoardLandmarks
    )
    from app.pipeline.debug import DebugContext
    from app.pipeline.debug.segmentation_debug import debug_segmentation

    # Build prompt
    has_prompt = False

    # Check for landmarks
    landmarks = None
    if any([args.nose, args.tail, args.rail_left, args.rail_right, args.center]):
        landmarks = BoardLandmarks(
            nose=parse_point(args.nose) if args.nose else None,
            tail=parse_point(args.tail) if args.tail else None,
            rail_left=parse_point(args.rail_left) if args.rail_left else None,
            rail_right=parse_point(args.rail_right) if args.rail_right else None,
            center=parse_point(args.center) if args.center else None,
        )
        has_prompt = True
        print(f"Landmarks: {landmarks.get_points()}")

    # Check for bbox
    bbox = None
    if args.bbox:
        x_min, y_min, x_max, y_max = parse_bbox(args.bbox)
        bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        has_prompt = True
        print(f"Bounding box: ({x_min:.2f}, {y_min:.2f}) to ({x_max:.2f}, {y_max:.2f})")

    # Check for click point
    click_point = None
    if args.click_x is not None and args.click_y is not None:
        click_point = ClickPoint(image_index=args.click_image, x=args.click_x, y=args.click_y)
        has_prompt = True
        print(f"Click point: image {args.click_image}, ({args.click_x:.2f}, {args.click_y:.2f})")

    # Extra points
    extra_fg = [parse_point(p) for p in (args.extra_fg or [])]
    extra_bg = [parse_point(p) for p in (args.extra_bg or [])]
    if extra_fg:
        has_prompt = True
        print(f"Extra foreground points: {extra_fg}")
    if extra_bg:
        print(f"Background points: {extra_bg}")

    # Default: center click
    if not has_prompt:
        print("No prompts specified, using center click as default")
        click_point = ClickPoint(image_index=0, x=0.5, y=0.5)

    # Create debug context
    debug = DebugContext(args.output_dir, enabled=True)
    print(f"Debug output: {args.output_dir}")

    # Load images
    print("\n=== Preprocessing ===")
    images, intrinsics = await preprocess_images(image_paths)
    print(f"Loaded {len(images)} images ({images[0].shape[1]}x{images[0].shape[0]})")

    # Build full prompt if we have more than just a click point
    if landmarks or bbox or extra_fg or extra_bg:
        prompt = SegmentationPrompt(
            image_index=args.click_image,
            click_point=click_point,
            landmarks=landmarks,
            extra_points=extra_fg,
            background_points=extra_bg,
            bounding_box=bbox,
        )
        use_full_prompt = True
    else:
        prompt = click_point
        use_full_prompt = False

    # Run segmentation
    print("\n=== Running SAM Segmentation ===")
    print(f"Prompt type: {'SegmentationPrompt' if use_full_prompt else 'ClickPoint'}")

    if args.refine > 0:
        print(f"With {args.refine} refinement iteration(s)")
        masks = await segment_board_with_refinement(images, prompt, args.refine)
    else:
        masks = await segment_board(images, prompt)

    print(f"Generated {len(masks)} masks")

    # Stats
    for i, mask in enumerate(masks):
        area_pct = (mask > 127).sum() / mask.size * 100
        print(f"  Mask {i}: {area_pct:.1f}% coverage")

    # Generate debug output
    print("\n=== Generating Debug Visualizations ===")

    # Determine click point for visualization
    if click_point:
        vis_click = (click_point.x, click_point.y)
        vis_idx = click_point.image_index
    elif landmarks and landmarks.center:
        vis_click = landmarks.center
        vis_idx = args.click_image
    else:
        vis_click = (0.5, 0.5)
        vis_idx = 0

    debug_segmentation(debug, images, masks, vis_click, vis_idx)

    # Additional visualizations for prompts
    stage_dir = debug.stage_dir("segmentation")

    # Visualize all prompt points
    for i, (img, mask) in enumerate(zip(images, masks)):
        vis = img.copy()
        h, w = vis.shape[:2]

        # Draw mask overlay
        overlay = vis.copy()
        overlay[mask > 127] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        # Draw bounding box
        if bbox:
            x1, y1, x2, y2 = bbox.to_pixels(w, h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(vis, "bbox", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw landmark points
        if landmarks:
            point_colors = {
                'nose': (255, 0, 0),      # Blue
                'tail': (0, 0, 255),       # Red
                'rail_left': (255, 255, 0), # Cyan
                'rail_right': (255, 0, 255), # Magenta
                'center': (0, 255, 0),     # Green
            }
            for name, pt in [
                ('nose', landmarks.nose),
                ('tail', landmarks.tail),
                ('rail_left', landmarks.rail_left),
                ('rail_right', landmarks.rail_right),
                ('center', landmarks.center),
            ]:
                if pt:
                    px, py = int(pt[0] * w), int(pt[1] * h)
                    color = point_colors[name]
                    cv2.circle(vis, (px, py), 8, color, -1)
                    cv2.circle(vis, (px, py), 10, (255, 255, 255), 2)
                    cv2.putText(vis, name, (px + 12, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw extra foreground points
        for pt in extra_fg:
            px, py = int(pt[0] * w), int(pt[1] * h)
            cv2.circle(vis, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(vis, "+", (px - 4, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw background points
        for pt in extra_bg:
            px, py = int(pt[0] * w), int(pt[1] * h)
            cv2.circle(vis, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(vis, "-", (px - 4, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw click point
        if click_point and i == click_point.image_index:
            px, py = int(click_point.x * w), int(click_point.y * h)
            cv2.drawMarker(vis, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        cv2.imwrite(str(stage_dir / f"prompts_{i:03d}.jpg"), vis)

    print(f"\nDone! Check output at: {args.output_dir}/02_segmentation/")
    print("\nKey files:")
    print(f"  - mask_overlay_*.jpg  : Mask overlays")
    print(f"  - prompts_*.jpg       : Shows all prompt points/boxes")
    print(f"  - mask_comparison.jpg : Side-by-side comparison")
    print(f"  - mask_stats.json     : Coverage stats")


if __name__ == "__main__":
    asyncio.run(main())
