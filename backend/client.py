#!/usr/bin/env python3
"""
Test client for the Surfboard Volume Calculator API.

Usage:
    python client.py <image_dir> [--url URL] [--scale-method METHOD] [--length MM] [--width MM] [--thickness MM]

Examples:
    # Using user-provided dimensions
    python client.py ./surfboard_photos --length 1830 --width 520 --thickness 63

    # Using ArUco marker (ID 42, 50mm size)
    python client.py ./surfboard_photos --scale-method aruco --aruco-id 42 --aruco-size 50

    # Using credit card for scale
    python client.py ./surfboard_photos --scale-method credit_card
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def get_image_files(directory: Path) -> list[Path]:
    """Get all image files from a directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(files)


def create_job(
    api_url: str,
    image_files: list[Path],
    scale_method: str,
    aruco_id: int | None = None,
    aruco_size: float | None = None,
    length_mm: float | None = None,
    width_mm: float | None = None,
    thickness_mm: float | None = None,
    click_image_index: int = 0,
    click_x: float = 0.5,
    click_y: float = 0.5,
) -> str:
    """Create a new volume calculation job."""

    # Build request data
    request_data = {
        "scale_method": scale_method,
        "board_click_point": {
            "image_index": click_image_index,
            "x": click_x,
            "y": click_y,
        }
    }

    if scale_method == "aruco":
        request_data["aruco_data"] = {
            "aruco_id": aruco_id or 42,
            "aruco_size_mm": aruco_size or 50.0,
        }
    elif scale_method == "credit_card":
        request_data["credit_card_data"] = {
            "card_type": "standard"
        }
    elif scale_method == "user_dimensions":
        request_data["user_dimensions"] = {}
        if length_mm:
            request_data["user_dimensions"]["length_mm"] = length_mm
        if width_mm:
            request_data["user_dimensions"]["width_mm"] = width_mm
        if thickness_mm:
            request_data["user_dimensions"]["thickness_mm"] = thickness_mm

    # Prepare files for upload
    files = []
    for img_path in image_files:
        files.append(
            ("images", (img_path.name, open(img_path, "rb"), "image/jpeg"))
        )

    try:
        response = requests.post(
            f"{api_url}/jobs",
            files=files,
            data={"request_data": json.dumps(request_data)},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["job_id"]
    finally:
        # Close file handles
        for _, (_, f, _) in files:
            f.close()


def poll_job(api_url: str, job_id: str, poll_interval: float = 2.0) -> dict:
    """Poll job status until completion."""
    print(f"Job created: {job_id}")
    print("Polling for results...")

    while True:
        response = requests.get(f"{api_url}/jobs/{job_id}")
        response.raise_for_status()
        data = response.json()

        status = data["status"]
        progress = data.get("progress", "")

        print(f"  Status: {status} - {progress}")

        if status in ("completed", "partial", "failed"):
            return data

        time.sleep(poll_interval)


def print_results(result: dict):
    """Pretty print the job results."""
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    status = result["status"]
    print(f"Status: {status}")

    if result.get("errors"):
        print("\nErrors:")
        for error in result["errors"]:
            print(f"  - {error}")

    if result.get("result"):
        r = result["result"]
        print(f"\nVolume: {r['volume_liters']:.2f} liters")
        print(f"\nDimensions:")
        print(f"  Length:    {r['dimensions']['length_mm']:.1f} mm")
        print(f"  Width:     {r['dimensions']['width_mm']:.1f} mm")
        print(f"  Thickness: {r['dimensions']['thickness_mm']:.1f} mm")
        print(f"\nConfidence: {r['confidence']:.1%}")

        if r.get("warnings"):
            print("\nWarnings:")
            for warning in r["warnings"]:
                print(f"  - {warning}")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Test client for Surfboard Volume Calculator API"
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing surfboard images"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--scale-method",
        choices=["aruco", "credit_card", "user_dimensions"],
        default="user_dimensions",
        help="Scale detection method"
    )
    parser.add_argument(
        "--aruco-id",
        type=int,
        default=42,
        help="ArUco marker ID (for aruco method)"
    )
    parser.add_argument(
        "--aruco-size",
        type=float,
        default=50.0,
        help="ArUco marker size in mm (for aruco method)"
    )
    parser.add_argument(
        "--length",
        type=float,
        help="Board length in mm"
    )
    parser.add_argument(
        "--width",
        type=float,
        help="Board width in mm"
    )
    parser.add_argument(
        "--thickness",
        type=float,
        help="Board thickness in mm"
    )
    parser.add_argument(
        "--click-image",
        type=int,
        default=0,
        help="Index of image where board is clicked (default: 0)"
    )
    parser.add_argument(
        "--click-x",
        type=float,
        default=0.5,
        help="Normalized X coordinate of click point (0-1, default: 0.5)"
    )
    parser.add_argument(
        "--click-y",
        type=float,
        default=0.5,
        help="Normalized Y coordinate of click point (0-1, default: 0.5)"
    )

    args = parser.parse_args()

    # Validate image directory
    if not args.image_dir.is_dir():
        print(f"Error: {args.image_dir} is not a directory")
        sys.exit(1)

    # Get image files
    image_files = get_image_files(args.image_dir)
    if len(image_files) < 3:
        print(f"Error: Need at least 3 images, found {len(image_files)}")
        sys.exit(1)

    print(f"Found {len(image_files)} images")

    # Validate scale method requirements
    if args.scale_method == "user_dimensions":
        if not any([args.length, args.width, args.thickness]):
            print("Warning: No dimensions provided. Results may be inaccurate.")
            print("Consider providing --length, --width, and/or --thickness")

    # Create and poll job
    try:
        job_id = create_job(
            api_url=args.url,
            image_files=image_files,
            scale_method=args.scale_method,
            aruco_id=args.aruco_id,
            aruco_size=args.aruco_size,
            length_mm=args.length,
            width_mm=args.width,
            thickness_mm=args.thickness,
            click_image_index=args.click_image,
            click_x=args.click_x,
            click_y=args.click_y,
        )

        result = poll_job(args.url, job_id)
        print_results(result)

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {args.url}")
        print("Make sure the server is running.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
