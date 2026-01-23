#!/usr/bin/env python3
"""
Video to Images Conversion Script

Extract optimal frames from video for photogrammetry reconstruction.
Supports both camera-moving and turntable (board-rotating) capture scenarios.

Usage:
    python video_to_images.py input.mp4 output_dir/
      --min-motion 0.02      # Minimum motion threshold (0-1)
      --min-sharpness 100    # Minimum Laplacian variance
      --max-frames 50        # Maximum frames to extract
      --format jpg|png       # Output format
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class FrameMetadata:
    """Metadata for an extracted frame."""
    filename: str
    frame_number: int
    timestamp_ms: float
    motion_score: float
    blur_score: float


@dataclass
class ExtractionResult:
    """Result of the video extraction process."""
    capture_mode: str  # "camera_moving" or "turntable"
    turntable_confidence: float
    total_frames_analyzed: int
    frames_extracted: int
    frames: list[FrameMetadata]
    warnings: list[str]


def calculate_blur_score(frame: np.ndarray) -> float:
    """
    Calculate blur score using Laplacian variance.
    Higher values indicate sharper images.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def calculate_motion_score(
    frame1: np.ndarray,
    frame2: np.ndarray,
    use_optical_flow: bool = True
) -> float:
    """
    Calculate motion score between two frames.
    Returns normalized value (0-1) indicating amount of change.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    if use_optical_flow:
        # Use Farneback optical flow for motion estimation
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        # Calculate magnitude of flow vectors
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        # Normalize by image diagonal
        diagonal = np.sqrt(frame1.shape[0]**2 + frame1.shape[1]**2)
        motion = np.mean(magnitude) / diagonal
    else:
        # Simple frame difference
        diff = cv2.absdiff(gray1, gray2)
        motion = np.mean(diff) / 255.0

    return float(min(1.0, motion * 10))  # Scale and clamp to 0-1


def analyze_edge_motion(
    frames: list[np.ndarray],
    edge_ratio: float = 0.15
) -> tuple[float, float]:
    """
    Analyze motion in edge regions vs center to detect turntable scenario.

    Returns:
        edge_motion: Average motion in edge regions
        center_motion: Average motion in center region
    """
    if len(frames) < 2:
        return 0.0, 0.0

    h, w = frames[0].shape[:2]
    edge_h = int(h * edge_ratio)
    edge_w = int(w * edge_ratio)

    edge_motions = []
    center_motions = []

    for i in range(1, min(len(frames), 10)):  # Sample first 10 frame pairs
        gray1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2).astype(float)

        # Edge regions (top, bottom, left, right strips)
        top = diff[:edge_h, :].mean()
        bottom = diff[-edge_h:, :].mean()
        left = diff[:, :edge_w].mean()
        right = diff[:, -edge_w:].mean()
        edge_motion = (top + bottom + left + right) / 4.0

        # Center region
        center = diff[edge_h:-edge_h, edge_w:-edge_w].mean()

        edge_motions.append(edge_motion)
        center_motions.append(center)

    return float(np.mean(edge_motions)), float(np.mean(center_motions))


def detect_capture_mode(
    sample_frames: list[np.ndarray]
) -> tuple[str, float]:
    """
    Detect whether video is camera-moving or turntable style.

    Returns:
        mode: "camera_moving" or "turntable"
        confidence: Confidence in the detection (0-1)
    """
    edge_motion, center_motion = analyze_edge_motion(sample_frames)

    # In turntable mode: edges are static, center has motion
    # In camera-moving mode: both edges and center have motion

    if edge_motion < 1.0 and center_motion > 2.0:
        # Static edges, moving center -> turntable
        ratio = center_motion / max(edge_motion, 0.1)
        confidence = min(1.0, ratio / 10.0)
        return "turntable", confidence
    elif edge_motion > 1.0:
        # Moving edges -> camera moving
        confidence = min(1.0, edge_motion / 5.0)
        return "camera_moving", confidence
    else:
        # Unclear, default to camera moving
        return "camera_moving", 0.5


def extract_frames(
    video_path: Path,
    output_dir: Path,
    min_motion: float = 0.02,
    min_sharpness: float = 100.0,
    max_frames: int = 50,
    output_format: str = "jpg",
    verbose: bool = True
) -> ExtractionResult:
    """
    Extract optimal frames from video for photogrammetry.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        min_motion: Minimum motion threshold (0-1) between frames
        min_sharpness: Minimum Laplacian variance for sharpness
        max_frames: Maximum number of frames to extract
        output_format: Output image format (jpg or png)
        verbose: Print progress information

    Returns:
        ExtractionResult with metadata about extracted frames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        print(f"Video: {video_path.name}")
        print(f"  FPS: {fps:.2f}, Total frames: {total_frames}")
        print(f"  Duration: {total_frames / fps:.1f}s")

    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Failed to read first frame")

    extracted_frames: list[FrameMetadata] = []
    sample_frames: list[np.ndarray] = [first_frame.copy()]
    warnings: list[str] = []

    reference_frame = first_frame.copy()
    last_selected_frame_num = 0
    frame_num = 0
    cumulative_motion = 0.0

    # Extension based on format
    ext = "." + output_format.lower()
    quality_params = [cv2.IMWRITE_JPEG_QUALITY, 95] if output_format.lower() == "jpg" else []

    # Calculate adaptive frame skip based on video length
    # For longer videos, we can skip more frames to save processing time
    frame_skip = max(1, total_frames // (max_frames * 20))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Skip frames for efficiency
        if frame_num % frame_skip != 0:
            continue

        # Collect sample frames for mode detection
        if len(sample_frames) < 10:
            sample_frames.append(frame.copy())

        # Calculate motion since last selected frame
        motion = calculate_motion_score(reference_frame, frame)
        cumulative_motion += motion

        # Check if we have enough motion
        if cumulative_motion >= min_motion:
            # Check sharpness
            blur_score = calculate_blur_score(frame)

            if blur_score >= min_sharpness:
                # Save frame
                timestamp_ms = (frame_num / fps) * 1000
                filename = f"frame_{len(extracted_frames):04d}{ext}"
                output_path = output_dir / filename

                cv2.imwrite(str(output_path), frame, quality_params)

                metadata = FrameMetadata(
                    filename=filename,
                    frame_number=frame_num,
                    timestamp_ms=timestamp_ms,
                    motion_score=cumulative_motion,
                    blur_score=blur_score
                )
                extracted_frames.append(metadata)

                if verbose:
                    print(f"  Extracted {filename} (motion={cumulative_motion:.3f}, blur={blur_score:.1f})")

                # Update reference
                reference_frame = frame.copy()
                cumulative_motion = 0.0
                last_selected_frame_num = frame_num

                # Check if we've reached max frames
                if len(extracted_frames) >= max_frames:
                    if verbose:
                        print(f"  Reached max frames limit ({max_frames})")
                    break
            else:
                if verbose and blur_score < min_sharpness * 0.5:
                    pass  # Too blurry, skip silently

    cap.release()

    # Detect capture mode from sampled frames
    capture_mode, turntable_confidence = detect_capture_mode(sample_frames)

    if capture_mode == "turntable" and turntable_confidence > 0.6:
        warnings.append(
            f"Turntable/rotating capture detected (confidence: {turntable_confidence:.2f}). "
            "The pipeline will use stricter background masking."
        )

    # Check for potential issues
    if len(extracted_frames) < 3:
        warnings.append(
            f"Only {len(extracted_frames)} frames extracted. "
            "Consider lowering --min-motion threshold or using a video with more movement."
        )

    if len(extracted_frames) > 0:
        avg_blur = np.mean([f.blur_score for f in extracted_frames])
        if avg_blur < min_sharpness * 1.5:
            warnings.append(
                f"Average sharpness is low ({avg_blur:.1f}). "
                "Consider using better lighting or a tripod."
            )

    result = ExtractionResult(
        capture_mode=capture_mode,
        turntable_confidence=turntable_confidence,
        total_frames_analyzed=frame_num,
        frames_extracted=len(extracted_frames),
        frames=extracted_frames,
        warnings=warnings
    )

    # Save metadata
    metadata_path = output_dir / "extraction_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    if verbose:
        print(f"\nExtraction complete:")
        print(f"  Frames extracted: {len(extracted_frames)}")
        print(f"  Capture mode: {capture_mode}")
        if capture_mode == "turntable":
            print(f"  Turntable confidence: {turntable_confidence:.2f}")
        print(f"  Metadata saved to: {metadata_path}")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract optimal frames from video for photogrammetry reconstruction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python video_to_images.py surfboard_video.mp4 ./frames/

  # Extract more frames with lower motion threshold
  python video_to_images.py video.mp4 ./output/ --min-motion 0.01 --max-frames 80

  # Higher quality PNG output
  python video_to_images.py video.mp4 ./output/ --format png --min-sharpness 150
        """
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input video file path"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for extracted frames"
    )
    parser.add_argument(
        "--min-motion",
        type=float,
        default=0.02,
        help="Minimum motion threshold (0-1) between selected frames (default: 0.02)"
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=100.0,
        help="Minimum Laplacian variance for frame sharpness (default: 100)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Maximum number of frames to extract (default: 50)"
    )
    parser.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="Output image format (default: jpg)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.input.is_file():
        print(f"Error: Input is not a file: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        result = extract_frames(
            video_path=args.input,
            output_dir=args.output,
            min_motion=args.min_motion,
            min_sharpness=args.min_sharpness,
            max_frames=args.max_frames,
            output_format=args.format,
            verbose=not args.quiet
        )

        if result.frames_extracted == 0:
            print("Error: No frames were extracted. Try lowering --min-motion or --min-sharpness.", file=sys.stderr)
            sys.exit(1)

        sys.exit(0)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
