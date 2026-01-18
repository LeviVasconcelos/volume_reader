"""Debug visualizations for the scale detection stage."""

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.pipeline.debug import DebugContext


def draw_aruco_detections(
    image: np.ndarray,
    corners: Optional[np.ndarray],
    ids: Optional[np.ndarray],
    target_id: Optional[int] = None,
) -> np.ndarray:
    """
    Draw detected ArUco markers on image.

    Args:
        image: Original image
        corners: Detected marker corners
        ids: Detected marker IDs
        target_id: Specific ID to highlight

    Returns:
        Annotated image
    """
    vis = image.copy()

    if ids is None or corners is None:
        cv2.putText(
            vis, "No ArUco markers detected",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
        )
        return vis

    for i, (marker_corners, marker_id) in enumerate(zip(corners, ids)):
        marker_id = int(marker_id[0])
        pts = marker_corners[0].astype(int)

        # Draw polygon
        color = (0, 255, 0) if target_id is None or marker_id == target_id else (255, 165, 0)
        cv2.polylines(vis, [pts], True, color, 3)

        # Draw corner points
        for j, pt in enumerate(pts):
            cv2.circle(vis, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(vis, str(j), tuple(pt + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw ID
        center = pts.mean(axis=0).astype(int)
        cv2.putText(
            vis, f"ID: {marker_id}",
            (center[0] - 30, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

        # Calculate and display size
        edge_lengths = [np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]
        avg_size = np.mean(edge_lengths)
        cv2.putText(
            vis, f"{avg_size:.1f}px",
            (center[0] - 30, center[1] + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

    return vis


def draw_card_detections(
    image: np.ndarray,
    detections: list[dict],
) -> np.ndarray:
    """
    Draw detected credit card rectangles on image.

    Args:
        image: Original image
        detections: List of detection dicts with 'corners' and 'confidence'

    Returns:
        Annotated image
    """
    vis = image.copy()

    if not detections:
        cv2.putText(
            vis, "No credit card detected",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
        )
        return vis

    for i, det in enumerate(detections):
        corners = det.get("corners")
        if corners is None:
            continue

        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 255), 3)

        # Draw corner points
        for j, pt in enumerate(pts):
            cv2.circle(vis, tuple(pt), 5, (0, 0, 255), -1)

        # Label
        center = pts.mean(axis=0).astype(int)
        cv2.putText(
            vis, f"Card {i+1}",
            (int(center[0]) - 30, int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

    return vis


def create_scale_consistency_plot(
    scales: list[float],
    method: str,
) -> np.ndarray:
    """
    Create plot showing scale consistency across images.

    Args:
        scales: Scale factors per image
        method: Scale detection method used

    Returns:
        Plot as image array
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        x = list(range(len(scales)))
        ax.bar(x, scales, color='steelblue', alpha=0.7)

        # Median line
        if scales:
            median = np.median(scales)
            ax.axhline(y=median, color='red', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')

            # Std dev band
            std = np.std(scales)
            ax.axhspan(median - std, median + std, color='red', alpha=0.1, label=f'Std: {std:.4f}')

        ax.set_xlabel('Image Index', fontsize=12)
        ax.set_ylabel('Scale Factor (mm/px)', fontsize=12)
        ax.set_title(f'Scale Consistency - {method}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        # Convert to image
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # Convert RGB to BGR
        return cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    except ImportError:
        # Fallback if matplotlib not available
        img = np.full((400, 800, 3), 255, dtype=np.uint8)
        cv2.putText(img, "matplotlib required for plots", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img


def debug_scale_detection(
    ctx: "DebugContext",
    images: list[np.ndarray],
    scale_factor: float,
    method: str,
    detections: list[dict],
) -> None:
    """
    Generate debug visualizations for the scale detection stage.

    Outputs:
        - aruco_detection_XXX.jpg or card_detection_XXX.jpg: Detection visualizations
        - scale_consistency.png: Plot of scale factors
        - scale_summary.json: Detection summary

    Args:
        ctx: Debug context
        images: Input images
        scale_factor: Computed scale factor
        method: Detection method ("aruco", "credit_card", or "user_dimensions")
        detections: Per-image detection results
    """
    if not ctx.enabled:
        return

    scales = []

    # Process each image
    for i, img in enumerate(images):
        det = detections[i] if i < len(detections) else {}

        if method == "aruco":
            # Detect ArUco markers for visualization
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
            aruco_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            vis = draw_aruco_detections(img, corners, ids, det.get("target_id"))
            ctx.save_image("scale", f"aruco_detection_{i:03d}", vis)

            if det.get("scale"):
                scales.append(det["scale"])

        elif method == "credit_card":
            card_dets = det.get("cards", [])
            vis = draw_card_detections(img, card_dets)
            ctx.save_image("scale", f"card_detection_{i:03d}", vis)

            if det.get("scale"):
                scales.append(det["scale"])

        else:
            # User dimensions - no per-image detection
            pass

    # Create scale consistency plot
    if scales:
        plot = create_scale_consistency_plot(scales, method)
        ctx.save_image("scale", "scale_consistency.png", plot)

    # Save summary
    summary = {
        "method": method,
        "final_scale_factor": scale_factor,
        "scale_unit": "mm per pixel",
        "num_detections": len(scales),
        "num_images": len(images),
        "scale_statistics": {
            "median": float(np.median(scales)) if scales else None,
            "mean": float(np.mean(scales)) if scales else None,
            "std": float(np.std(scales)) if scales else None,
            "min": float(np.min(scales)) if scales else None,
            "max": float(np.max(scales)) if scales else None,
        },
        "per_image_detections": detections,
    }
    ctx.save_json("scale", "scale_summary", summary)
