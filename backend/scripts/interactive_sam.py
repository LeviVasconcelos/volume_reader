#!/usr/bin/env python3
"""
Interactive SAM Segmentation Tool

A GUI tool for placing landmarks (bounding box and foreground points) on images,
then running SAM segmentation and viewing results.

Usage:
    python interactive_sam.py ./surfboard_frames/
    python interactive_sam.py ./surfboard_frames/ --output ./segmentation_output/
    python interactive_sam.py ./surfboard_frames/ --load annotations.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add backend directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from abc import ABC, abstractmethod
from enum import Enum

from app.models import BoundingBox
from app.pipeline.segmentation import segment_with_prompts


class FeatureType(Enum):
    """Available feature detector types."""
    SIFT = "sift"
    ORB = "orb"
    AKAZE = "akaze"


class FeatureDetector(ABC):
    """Abstract base class for feature detectors."""
    name: str  # For display

    @abstractmethod
    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None
               ) -> tuple[list[cv2.KeyPoint], Optional[np.ndarray]]:
        """Detect keypoints and compute descriptors in image, optionally restricted to mask region."""
        pass


class SIFTDetector(FeatureDetector):
    """SIFT feature detector."""
    name = "SIFT"

    def __init__(self, max_features: int = 2000):
        self.detector = cv2.SIFT_create(nfeatures=max_features)

    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None
               ) -> tuple[list[cv2.KeyPoint], Optional[np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask_u8 = mask.astype(np.uint8) if mask is not None else None
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask_u8)
        return (keypoints if keypoints else [], descriptors)


class ORBDetector(FeatureDetector):
    """ORB feature detector."""
    name = "ORB"

    def __init__(self, max_features: int = 2000):
        self.detector = cv2.ORB_create(nfeatures=max_features)

    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None
               ) -> tuple[list[cv2.KeyPoint], Optional[np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask_u8 = mask.astype(np.uint8) if mask is not None else None
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask_u8)
        return (keypoints if keypoints else [], descriptors)


class AKAZEDetector(FeatureDetector):
    """AKAZE feature detector."""
    name = "AKAZE"

    def __init__(self):
        self.detector = cv2.AKAZE_create()

    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None
               ) -> tuple[list[cv2.KeyPoint], Optional[np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask_u8 = mask.astype(np.uint8) if mask is not None else None
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask_u8)
        return (keypoints if keypoints else [], descriptors)


@dataclass
class ImageAnnotation:
    """Stores annotations for a single image."""
    foreground_points: list[tuple[float, float]] = field(default_factory=list)  # normalized (0-1)
    bounding_box: Optional[tuple[float, float, float, float]] = None  # x1,y1,x2,y2 normalized

    def to_dict(self) -> dict:
        return {
            "foreground_points": self.foreground_points,
            "bounding_box": list(self.bounding_box) if self.bounding_box else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageAnnotation":
        return cls(
            foreground_points=[(p[0], p[1]) for p in data.get("foreground_points", [])],
            bounding_box=tuple(data["bounding_box"]) if data.get("bounding_box") else None,
        )


class InteractiveSAM:
    """Interactive SAM segmentation tool with OpenCV GUI."""

    WINDOW_NAME = "Interactive SAM Segmentation"
    TRACKING_WINDOW_NAME = "Feature Tracking"
    MAX_DISPLAY_DIM = 1024  # Max dimension for display (keeps images manageable)
    POINT_RADIUS = 8
    POINT_COLOR = (0, 255, 0)  # Green (BGR)
    POINT_OUTLINE = (255, 255, 255)  # White
    BBOX_COLOR = (0, 255, 255)  # Yellow
    BBOX_PREVIEW_COLOR = (0, 200, 200)  # Darker yellow for preview
    MASK_COLOR = (0, 255, 0)  # Green for mask overlay
    MASK_ALPHA = 0.4
    POINT_REMOVE_THRESHOLD = 20  # pixels

    def __init__(self, image_dir: Path, output_dir: Optional[Path] = None):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.image_paths: list[Path] = []
        self.images: list[np.ndarray] = []  # Original full-res images (for SAM)
        self.display_images: list[np.ndarray] = []  # Scaled images for display
        self.annotations: dict[int, ImageAnnotation] = {}
        self.masks: dict[int, np.ndarray] = {}
        self.keypoints: dict[int, list[cv2.KeyPoint]] = {}  # Per-image keypoints
        self.descriptors: dict[int, np.ndarray] = {}  # Per-image descriptors
        self.feature_detector: FeatureDetector = SIFTDetector()  # Current detector
        self.show_keypoints: bool = True  # Toggle visibility
        self.tracking_mode: bool = False  # Toggle for feature tracking

        self.current_idx = 0
        self.mode = "point"  # 'point' or 'bbox'
        self.bbox_start: Optional[tuple[int, int]] = None  # temp storage during drag
        self.is_dragging = False
        self.current_mouse_pos: tuple[int, int] = (0, 0)

        self._load_images()
        self._setup_window()

    def _load_images(self) -> None:
        """Load all images from directory."""
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        self.image_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in extensions
        )

        if not self.image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"Loading {len(self.image_paths)} images...")
        for path in self.image_paths:
            img = cv2.imread(str(path))
            if img is None:
                print(f"Warning: Could not load {path}")
                continue
            self.images.append(img)
            self.display_images.append(self._resize_for_display(img))

        print(f"Loaded {len(self.images)} images successfully")
        if self.images:
            orig_h, orig_w = self.images[0].shape[:2]
            disp_h, disp_w = self.display_images[0].shape[:2]
            print(f"Original size: {orig_w}x{orig_h}, Display size: {disp_w}x{disp_h}")

    def _resize_for_display(self, img: np.ndarray) -> np.ndarray:
        """Resize image for display if it exceeds MAX_DISPLAY_DIM."""
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self.MAX_DISPLAY_DIM:
            return img.copy()
        scale = self.MAX_DISPLAY_DIM / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _setup_window(self) -> None:
        """Create OpenCV window and set up mouse callback."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)

    def _get_annotation(self, idx: int) -> ImageAnnotation:
        """Get or create annotation for image index."""
        if idx not in self.annotations:
            self.annotations[idx] = ImageAnnotation()
        return self.annotations[idx]

    def _pixel_to_normalized(self, x: int, y: int) -> tuple[float, float]:
        """Convert display pixel coordinates to normalized (0-1)."""
        img = self.display_images[self.current_idx]
        h, w = img.shape[:2]
        return (x / w, y / h)

    def _normalized_to_pixel(self, nx: float, ny: float) -> tuple[int, int]:
        """Convert normalized coordinates to display pixels."""
        img = self.display_images[self.current_idx]
        h, w = img.shape[:2]
        return (int(nx * w), int(ny * h))

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events."""
        self.current_mouse_pos = (x, y)
        annotation = self._get_annotation(self.current_idx)

        if self.mode == "point":
            self._handle_point_mode(event, x, y, annotation)
        else:  # bbox mode
            self._handle_bbox_mode(event, x, y, flags, annotation)

        self._render()

    def _handle_point_mode(self, event: int, x: int, y: int, annotation: ImageAnnotation) -> None:
        """Handle mouse events in point mode."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add foreground point
            norm_point = self._pixel_to_normalized(x, y)
            annotation.foreground_points.append(norm_point)
            print(f"Added point at ({norm_point[0]:.3f}, {norm_point[1]:.3f})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove nearest point or clear bbox
            if annotation.foreground_points:
                nearest_idx, dist = self._find_nearest_point(x, y, annotation.foreground_points)
                if dist < self.POINT_REMOVE_THRESHOLD:
                    removed = annotation.foreground_points.pop(nearest_idx)
                    print(f"Removed point at ({removed[0]:.3f}, {removed[1]:.3f})")
                    return
            # If no point removed, clear bbox
            if annotation.bounding_box:
                annotation.bounding_box = None
                print("Cleared bounding box")

    def _handle_bbox_mode(self, event: int, x: int, y: int, flags: int, annotation: ImageAnnotation) -> None:
        """Handle mouse events in bbox mode."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.bbox_start = (x, y)
            self.is_dragging = True

        elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
            # Preview is drawn in render
            pass

        elif event == cv2.EVENT_LBUTTONUP and self.is_dragging:
            if self.bbox_start:
                x1, y1 = self.bbox_start
                x2, y2 = x, y
                # Ensure proper ordering
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                # Convert to normalized
                norm_x1, norm_y1 = self._pixel_to_normalized(x1, y1)
                norm_x2, norm_y2 = self._pixel_to_normalized(x2, y2)
                annotation.bounding_box = (norm_x1, norm_y1, norm_x2, norm_y2)
                print(f"Set bbox: ({norm_x1:.3f}, {norm_y1:.3f}) to ({norm_x2:.3f}, {norm_y2:.3f})")
            self.bbox_start = None
            self.is_dragging = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click clears bbox or removes nearest point
            if annotation.bounding_box:
                annotation.bounding_box = None
                print("Cleared bounding box")
            elif annotation.foreground_points:
                nearest_idx, dist = self._find_nearest_point(x, y, annotation.foreground_points)
                if dist < self.POINT_REMOVE_THRESHOLD:
                    removed = annotation.foreground_points.pop(nearest_idx)
                    print(f"Removed point at ({removed[0]:.3f}, {removed[1]:.3f})")

    def _find_nearest_point(self, x: int, y: int, points: list[tuple[float, float]]) -> tuple[int, float]:
        """Find nearest point to pixel coordinates. Returns (index, distance)."""
        min_dist = float("inf")
        min_idx = 0
        for i, (nx, ny) in enumerate(points):
            px, py = self._normalized_to_pixel(nx, ny)
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        return min_idx, min_dist

    def _render(self) -> None:
        """Render current frame with all overlays."""
        img = self.display_images[self.current_idx].copy()
        annotation = self._get_annotation(self.current_idx)

        # Draw mask overlay if exists (resize mask to display size)
        if self.current_idx in self.masks:
            mask = self.masks[self.current_idx]
            disp_h, disp_w = img.shape[:2]
            mask_resized = cv2.resize(mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            overlay = np.zeros_like(img)
            overlay[mask_resized > 0] = self.MASK_COLOR
            img = cv2.addWeighted(img, 1.0, overlay, self.MASK_ALPHA, 0)

        # Draw bounding box
        if annotation.bounding_box:
            x1, y1, x2, y2 = annotation.bounding_box
            px1, py1 = self._normalized_to_pixel(x1, y1)
            px2, py2 = self._normalized_to_pixel(x2, y2)
            cv2.rectangle(img, (px1, py1), (px2, py2), self.BBOX_COLOR, 2)

        # Draw bbox preview during drag
        if self.is_dragging and self.bbox_start:
            x1, y1 = self.bbox_start
            x2, y2 = self.current_mouse_pos
            # Draw dashed rectangle (simulate with dotted line)
            self._draw_dashed_rect(img, (x1, y1), (x2, y2), self.BBOX_PREVIEW_COLOR, 2)

        # Draw foreground points
        for nx, ny in annotation.foreground_points:
            px, py = self._normalized_to_pixel(nx, ny)
            cv2.circle(img, (px, py), self.POINT_RADIUS, self.POINT_OUTLINE, -1)
            cv2.circle(img, (px, py), self.POINT_RADIUS - 2, self.POINT_COLOR, -1)

        # Draw keypoints
        if self.show_keypoints and self.current_idx in self.keypoints:
            self._draw_keypoints(img)

        # Draw status bar (top)
        h, w = img.shape[:2]
        status_height = 30
        cv2.rectangle(img, (0, 0), (w, status_height), (40, 40, 40), -1)
        has_bbox = "Yes" if annotation.bounding_box else "No"
        has_mask = " | MASK" if self.current_idx in self.masks else ""
        kp_info = ""
        if self.current_idx in self.keypoints:
            kp_count = len(self.keypoints[self.current_idx])
            kp_info = f" | KP: {kp_count} ({self.feature_detector.name})"
        tracking_status = " | TRACK" if self.tracking_mode else ""
        status_text = f"Mode: {self.mode.upper()}{tracking_status} | Image {self.current_idx + 1}/{len(self.images)} | Points: {len(annotation.foreground_points)} | Bbox: {has_bbox}{has_mask}{kp_info}"
        cv2.putText(img, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw help bar (bottom)
        help_height = 25
        cv2.rectangle(img, (0, h - help_height), (w, h), (40, 40, 40), -1)
        help_text = "p:point b:bbox c:clear r:run f:features t:track k:kp-toggle q:quit"
        cv2.putText(img, help_text, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow(self.WINDOW_NAME, img)

    def _draw_dashed_rect(self, img: np.ndarray, pt1: tuple, pt2: tuple, color: tuple, thickness: int) -> None:
        """Draw a dashed rectangle."""
        x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
        x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
        dash_length = 10
        gap_length = 5

        # Draw each edge with dashes
        for start, end in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                           ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
            self._draw_dashed_line(img, start, end, color, thickness, dash_length, gap_length)

    def _draw_dashed_line(self, img: np.ndarray, pt1: tuple, pt2: tuple, color: tuple,
                          thickness: int, dash_length: int, gap_length: int) -> None:
        """Draw a dashed line."""
        dist = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        if dist == 0:
            return
        dx = (pt2[0] - pt1[0]) / dist
        dy = (pt2[1] - pt1[1]) / dist

        pos = 0
        drawing = True
        while pos < dist:
            segment_len = dash_length if drawing else gap_length
            end_pos = min(pos + segment_len, dist)
            if drawing:
                start_pt = (int(pt1[0] + pos * dx), int(pt1[1] + pos * dy))
                end_pt = (int(pt1[0] + end_pos * dx), int(pt1[1] + end_pos * dy))
                cv2.line(img, start_pt, end_pt, color, thickness)
            pos = end_pos
            drawing = not drawing

    def _draw_keypoints(self, img: np.ndarray) -> None:
        """Draw keypoints on display image."""
        keypoints = self.keypoints.get(self.current_idx, [])
        if not keypoints:
            return

        # Scale keypoints from original to display size
        orig_h, orig_w = self.images[self.current_idx].shape[:2]
        disp_h, disp_w = img.shape[:2]
        scale_x, scale_y = disp_w / orig_w, disp_h / orig_h

        for kp in keypoints:
            x = int(kp.pt[0] * scale_x)
            y = int(kp.pt[1] * scale_y)
            cv2.circle(img, (x, y), 3, (255, 255, 0), -1)  # Cyan filled
            cv2.circle(img, (x, y), 3, (0, 0, 0), 1)       # Black outline

    def _run_segmentation(self) -> None:
        """Run SAM segmentation on all annotated images."""
        annotated_indices = [i for i, ann in self.annotations.items()
                            if ann.foreground_points or ann.bounding_box]

        if not annotated_indices:
            print("No annotated images to segment")
            return

        print(f"Running SAM on {len(annotated_indices)} annotated images...")

        for idx in annotated_indices:
            annotation = self.annotations[idx]
            image = self.images[idx]

            # Build bounding box if present
            bbox = None
            if annotation.bounding_box:
                x1, y1, x2, y2 = annotation.bounding_box
                bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)

            print(f"  Segmenting image {idx + 1}/{len(self.images)}...")

            try:
                mask = segment_with_prompts(
                    image=image,
                    foreground_points=annotation.foreground_points if annotation.foreground_points else None,
                    background_points=None,
                    bounding_box=bbox,
                )
                self.masks[idx] = mask
                print(f"    Success! Mask coverage: {np.sum(mask > 0) / mask.size * 100:.1f}%")
            except Exception as e:
                print(f"    Error: {e}")

        print("Segmentation complete!")
        self._render()

    def _detect_features(self) -> None:
        """Detect keypoints and descriptors for current image."""
        image = self.images[self.current_idx]
        mask = self.masks.get(self.current_idx)  # May be None

        print(f"Detecting {self.feature_detector.name} features...")
        keypoints, descriptors = self.feature_detector.detect(image, mask)

        self.keypoints[self.current_idx] = keypoints
        self.descriptors[self.current_idx] = descriptors if descriptors is not None else np.array([])

        print(f"  Found {len(keypoints)} keypoints" +
              (" (within mask)" if mask is not None else ""))
        self._render()

    def _scale_keypoints_to_display(self, idx: int) -> list[cv2.KeyPoint]:
        """Scale keypoints from original image coordinates to display coordinates."""
        keypoints = self.keypoints.get(idx, [])
        if not keypoints:
            return []

        orig_h, orig_w = self.images[idx].shape[:2]
        disp_h, disp_w = self.display_images[idx].shape[:2]
        scale_x, scale_y = disp_w / orig_w, disp_h / orig_h

        scaled = []
        for kp in keypoints:
            new_kp = cv2.KeyPoint(
                x=kp.pt[0] * scale_x,
                y=kp.pt[1] * scale_y,
                size=kp.size * (scale_x + scale_y) / 2,
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id
            )
            scaled.append(new_kp)
        return scaled

    def _track_features(self, from_idx: int, to_idx: int) -> None:
        """Track features from one image to another and display results."""
        # Check if both images have descriptors
        if from_idx not in self.descriptors or to_idx not in self.descriptors:
            print(f"Cannot track: missing features for image {from_idx+1} or {to_idx+1}")
            print("  Run feature detection (f) on both images first")
            return

        desc1 = self.descriptors[from_idx]
        desc2 = self.descriptors[to_idx]

        if desc1 is None or desc2 is None or desc1.size == 0 or desc2.size == 0:
            print("Cannot track: one or both images have no descriptors")
            return

        # Check for SIFT (FLANN requires float descriptors)
        if not isinstance(self.feature_detector, SIFTDetector):
            print("Warning: Tracking requires SIFT features (FLANN uses float descriptors)")
            print("  Switch to SIFT (press 1) and re-detect features")
            return

        # Import matching functions from pipeline
        from app.pipeline.features import match_features, geometric_verification

        print(f"Tracking features: image {from_idx+1} -> {to_idx+1}...")

        # Match features
        matches = match_features(desc1, desc2, ratio_threshold=0.75)
        print(f"  Raw matches: {len(matches)}")

        # Geometric verification
        if len(matches) >= 8:
            kp1 = self.keypoints[from_idx]
            kp2 = self.keypoints[to_idx]
            inlier_matches, _ = geometric_verification(kp1, kp2, matches, ransac_threshold=4.0)
            print(f"  Inlier matches: {len(inlier_matches)}")
        else:
            inlier_matches = matches
            print("  Too few matches for geometric verification")

        # Create visualization
        from app.pipeline.debug.features_debug import draw_feature_matches

        img1 = self.display_images[from_idx]
        img2 = self.display_images[to_idx]
        kp1_scaled = self._scale_keypoints_to_display(from_idx)
        kp2_scaled = self._scale_keypoints_to_display(to_idx)

        vis = draw_feature_matches(img1, img2, kp1_scaled, kp2_scaled, inlier_matches, max_draw=100)

        # Add status bar to visualization
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 30), (40, 40, 40), -1)
        status = f"Image {from_idx+1} -> {to_idx+1} | Matches: {len(inlier_matches)}/{len(matches)} (inliers/total)"
        cv2.putText(vis, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(self.TRACKING_WINDOW_NAME, vis)

    def _save_annotations(self, filepath: Optional[Path] = None) -> None:
        """Save annotations to JSON file."""
        if filepath is None:
            filepath = self.image_dir / "annotations.json"

        data = {
            "image_files": [p.name for p in self.image_paths],
            "annotations": {
                str(idx): ann.to_dict()
                for idx, ann in self.annotations.items()
            }
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved annotations to {filepath}")

    def _load_annotations(self, filepath: Optional[Path] = None) -> None:
        """Load annotations from JSON file."""
        if filepath is None:
            filepath = self.image_dir / "annotations.json"

        if not filepath.exists():
            print(f"No annotations file found at {filepath}")
            return

        with open(filepath) as f:
            data = json.load(f)

        # Validate image files match
        saved_files = data.get("image_files", [])
        current_files = [p.name for p in self.image_paths]

        if saved_files != current_files:
            print("Warning: Image files don't match saved annotations")
            print(f"  Saved: {len(saved_files)} files")
            print(f"  Current: {len(current_files)} files")

        self.annotations = {
            int(idx): ImageAnnotation.from_dict(ann_data)
            for idx, ann_data in data.get("annotations", {}).items()
        }

        print(f"Loaded annotations for {len(self.annotations)} images from {filepath}")
        self._render()

    def _save_masks(self) -> None:
        """Save segmentation masks to output directory."""
        if self.output_dir is None:
            self.output_dir = self.image_dir / "masks"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for idx, mask in self.masks.items():
            filename = self.image_paths[idx].stem + "_mask.png"
            output_path = self.output_dir / filename
            cv2.imwrite(str(output_path), mask)
            print(f"Saved mask: {output_path}")

        print(f"Saved {len(self.masks)} masks to {self.output_dir}")

    def _navigate(self, delta: int) -> None:
        """Navigate to a different image."""
        new_idx = (self.current_idx + delta) % len(self.images)
        if new_idx != self.current_idx:
            old_idx = self.current_idx
            self.current_idx = new_idx
            self.bbox_start = None
            self.is_dragging = False

            # Track features if tracking mode is on
            if self.tracking_mode:
                self._track_features(old_idx, new_idx)

            self._render()

    def _clear_current(self) -> None:
        """Clear annotations for current image."""
        self.annotations[self.current_idx] = ImageAnnotation()
        if self.current_idx in self.masks:
            del self.masks[self.current_idx]
        if self.current_idx in self.keypoints:
            del self.keypoints[self.current_idx]
        if self.current_idx in self.descriptors:
            del self.descriptors[self.current_idx]
        print(f"Cleared annotations for image {self.current_idx + 1}")
        self._render()

    def run(self, load_file: Optional[Path] = None) -> None:
        """Main event loop."""
        if load_file:
            self._load_annotations(load_file)

        self._render()

        print("\n=== Interactive SAM Segmentation ===")
        print("Controls:")
        print("  Left/Right arrows: Navigate images")
        print("  p: Point mode")
        print("  b: Bbox mode")
        print("  c: Clear current image")
        print("  r/Enter: Run SAM segmentation")
        print("  s: Save annotations")
        print("  l: Load annotations")
        print("  m: Save masks (after segmentation)")
        print("  f: Detect features (keypoints)")
        print("  t: Toggle tracking mode")
        print("  k: Toggle keypoint visibility")
        print("  1/2/3: Switch detector (SIFT/ORB/AKAZE)")
        print("  q/Esc: Quit")
        print("=====================================\n")

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord("q") or key == 27:  # q or Esc
                break
            elif key == ord("p"):
                self.mode = "point"
                print("Switched to POINT mode")
                self._render()
            elif key == ord("b"):
                self.mode = "bbox"
                print("Switched to BBOX mode")
                self._render()
            elif key == ord("c"):
                self._clear_current()
            elif key == ord("r") or key == 13:  # r or Enter
                self._run_segmentation()
            elif key == ord("s"):
                self._save_annotations()
            elif key == ord("l"):
                self._load_annotations()
            elif key == ord("m"):
                self._save_masks()
            elif key == ord("f"):
                self._detect_features()
            elif key == ord("t"):
                self.tracking_mode = not self.tracking_mode
                print(f"Tracking mode: {'ON' if self.tracking_mode else 'OFF'}")
                if not self.tracking_mode:
                    try:
                        cv2.destroyWindow(self.TRACKING_WINDOW_NAME)
                    except cv2.error:
                        pass  # Window may not exist
                self._render()
            elif key == ord("k"):
                self.show_keypoints = not self.show_keypoints
                print(f"Keypoints: {'ON' if self.show_keypoints else 'OFF'}")
                self._render()
            elif key == ord("1"):
                self.feature_detector = SIFTDetector()
                print("Switched to SIFT detector")
            elif key == ord("2"):
                self.feature_detector = ORBDetector()
                print("Switched to ORB detector")
            elif key == ord("3"):
                self.feature_detector = AKAZEDetector()
                print("Switched to AKAZE detector")
            elif key == 81 or key == 2:  # Left arrow
                self._navigate(-1)
            elif key == 83 or key == 3:  # Right arrow
                self._navigate(1)

        try:
            cv2.destroyWindow(self.TRACKING_WINDOW_NAME)
        except cv2.error:
            pass
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive SAM segmentation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python interactive_sam.py ./surfboard_frames/
    python interactive_sam.py ./surfboard_frames/ --output ./segmentation_output/
    python interactive_sam.py ./surfboard_frames/ --load annotations.json
        """
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing images to segment"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for saved masks"
    )
    parser.add_argument(
        "--load", "-l",
        type=Path,
        default=None,
        help="Load annotations from JSON file"
    )

    args = parser.parse_args()

    if not args.image_dir.exists():
        print(f"Error: Image directory not found: {args.image_dir}")
        sys.exit(1)

    if not args.image_dir.is_dir():
        print(f"Error: Not a directory: {args.image_dir}")
        sys.exit(1)

    try:
        tool = InteractiveSAM(args.image_dir, args.output)
        tool.run(load_file=args.load)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
