#!/usr/bin/env python3
"""
Debug script for SAM segmentation only.

Usage:
    python debug_sam.py ./frames/ ./debug_sam_output/ [--click-image N] [--click-x X] [--click-y Y]
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def main():
    parser = argparse.ArgumentParser(description="Debug SAM segmentation")
    parser.add_argument("image_dir", type=Path, help="Directory containing images")
    parser.add_argument("output_dir", type=Path, help="Debug output directory")
    parser.add_argument("--click-image", type=int, default=0, help="Image index for click")
    parser.add_argument("--click-x", type=float, default=0.5, help="Normalized X (0-1)")
    parser.add_argument("--click-y", type=float, default=0.5, help="Normalized Y (0-1)")
    parser.add_argument("--max-images", type=int, help="Limit number of images")

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
    from app.pipeline.preprocess import preprocess_images
    from app.pipeline.segmentation import segment_board
    from app.models import ClickPoint
    from app.pipeline.debug import DebugContext
    from app.pipeline.debug.segmentation_debug import debug_segmentation

    # Create debug context
    debug = DebugContext(args.output_dir, enabled=True)
    print(f"Debug output: {args.output_dir}")

    # Stage 1: Preprocess
    print("\n=== Preprocessing ===")
    images, intrinsics = await preprocess_images(image_paths)
    print(f"Loaded {len(images)} images ({images[0].shape[1]}x{images[0].shape[0]})")

    # Show click point info
    click_img_idx = min(args.click_image, len(images) - 1)
    h, w = images[click_img_idx].shape[:2]
    click_px_x = int(args.click_x * w)
    click_px_y = int(args.click_y * h)
    print(f"\nClick point: image {click_img_idx}, ({args.click_x:.2f}, {args.click_y:.2f}) = pixel ({click_px_x}, {click_px_y})")

    # Stage 2: Segmentation
    print("\n=== Running SAM Segmentation ===")
    click_point = ClickPoint(
        image_index=click_img_idx,
        x=args.click_x,
        y=args.click_y
    )

    try:
        masks = await segment_board(images, click_point)
        print(f"Generated {len(masks)} masks")

        # Stats
        for i, mask in enumerate(masks):
            area_pct = (mask > 127).sum() / mask.size * 100
            print(f"  Mask {i}: {area_pct:.1f}% coverage")

    except Exception as e:
        print(f"Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate debug output
    print("\n=== Generating Debug Visualizations ===")
    debug_segmentation(debug, images, masks, (args.click_x, args.click_y), click_img_idx)

    print(f"\nDone! Check output at: {args.output_dir}/02_segmentation/")
    print("\nKey files:")
    print(f"  - mask_overlay_*.jpg  : Each image with green mask overlay")
    print(f"  - click_point.jpg     : Shows where the click point landed")
    print(f"  - mask_comparison.jpg : Grid comparing all masks")
    print(f"  - mask_stats.json     : Coverage percentages")


if __name__ == "__main__":
    asyncio.run(main())
