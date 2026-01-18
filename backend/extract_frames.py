#!/usr/bin/env python3
"""
Extract frames from a video file using OpenCV.

Usage:
    python extract_frames.py input.mp4 output_dir/ [--every N] [--fps F] [--max M]
"""

import argparse
import sys
from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    every_n_frames: int = None,
    target_fps: float = None,
    max_frames: int = None,
):
    """
    Extract frames from video.

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        every_n_frames: Extract every Nth frame (e.g., 30 = every 30th frame)
        target_fps: Target frames per second to extract
        max_frames: Maximum number of frames to extract
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"Video: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  Duration: {duration:.1f}s ({total_frames} frames)")

    # Determine frame interval
    if target_fps and video_fps > 0:
        every_n_frames = max(1, int(video_fps / target_fps))
    elif every_n_frames is None:
        # Default: extract ~1 fps
        every_n_frames = max(1, int(video_fps))

    expected_frames = total_frames // every_n_frames
    if max_frames:
        expected_frames = min(expected_frames, max_frames)

    print(f"  Extracting every {every_n_frames} frames (~{expected_frames} total)")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            output_path = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

            if saved_count % 10 == 0:
                print(f"  Saved {saved_count} frames...", end='\r')

            if max_frames and saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()

    print(f"\nExtracted {saved_count} frames to {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract 1 frame per second (default)
    python extract_frames.py surfboard.mp4 ./frames/

    # Extract every 15th frame
    python extract_frames.py surfboard.mp4 ./frames/ --every 15

    # Extract at 2 fps
    python extract_frames.py surfboard.mp4 ./frames/ --fps 2

    # Extract max 50 frames
    python extract_frames.py surfboard.mp4 ./frames/ --max 50
        """
    )

    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("output_dir", type=Path, help="Output directory for frames")
    parser.add_argument("--every", type=int, help="Extract every Nth frame")
    parser.add_argument("--fps", type=float, help="Target frames per second")
    parser.add_argument("--max", type=int, help="Maximum frames to extract")

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    extract_frames(
        args.video,
        args.output_dir,
        every_n_frames=args.every,
        target_fps=args.fps,
        max_frames=args.max,
    )


if __name__ == "__main__":
    main()
