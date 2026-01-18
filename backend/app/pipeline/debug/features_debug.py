"""Debug visualizations for the feature extraction and matching stage."""

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.pipeline.debug import DebugContext


def draw_keypoints_with_scale(
    image: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    max_draw: int = 2000,
) -> np.ndarray:
    """
    Draw keypoints colored by scale.

    Args:
        image: Original image
        keypoints: Detected keypoints
        max_draw: Maximum keypoints to draw

    Returns:
        Annotated image
    """
    vis = image.copy()

    if not keypoints:
        cv2.putText(vis, "No keypoints", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis

    # Get scale range for coloring
    scales = [kp.size for kp in keypoints]
    min_scale = min(scales)
    max_scale = max(scales)
    scale_range = max_scale - min_scale if max_scale > min_scale else 1.0

    # Sort by scale for consistent drawing order
    sorted_kps = sorted(keypoints, key=lambda kp: kp.size)

    # Sample if too many
    if len(sorted_kps) > max_draw:
        indices = np.linspace(0, len(sorted_kps) - 1, max_draw, dtype=int)
        sorted_kps = [sorted_kps[i] for i in indices]

    for kp in sorted_kps:
        # Color by scale (blue=small, red=large)
        t = (kp.size - min_scale) / scale_range
        color = (
            int(255 * (1 - t)),  # B
            int(255 * (1 - abs(2 * t - 1))),  # G
            int(255 * t),  # R
        )

        pt = (int(kp.pt[0]), int(kp.pt[1]))
        radius = max(2, int(kp.size / 4))
        cv2.circle(vis, pt, radius, color, 1)

    # Add legend
    cv2.putText(vis, f"Keypoints: {len(keypoints)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, "Blue=small, Red=large", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def create_keypoint_heatmap(
    image_shape: tuple[int, int],
    keypoints: list[cv2.KeyPoint],
    cell_size: int = 32,
) -> np.ndarray:
    """
    Create density heatmap of keypoint locations.

    Args:
        image_shape: (height, width)
        keypoints: Detected keypoints
        cell_size: Size of heatmap cells

    Returns:
        Heatmap image
    """
    h, w = image_shape
    grid_h = (h + cell_size - 1) // cell_size
    grid_w = (w + cell_size - 1) // cell_size

    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)

    for kp in keypoints:
        gx = min(int(kp.pt[0] / cell_size), grid_w - 1)
        gy = min(int(kp.pt[1] / cell_size), grid_h - 1)
        heatmap[gy, gx] += 1

    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Apply colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Resize to original image size
    heatmap_color = cv2.resize(heatmap_color, (w, h), interpolation=cv2.INTER_NEAREST)

    return heatmap_color


def create_match_matrix_plot(
    n_images: int,
    matches: dict[tuple[int, int], list[cv2.DMatch]],
) -> np.ndarray:
    """
    Create heatmap of match counts between image pairs.

    Args:
        n_images: Number of images
        matches: Dictionary of matches per pair

    Returns:
        Match matrix plot as image
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Build matrix
        matrix = np.zeros((n_images, n_images))
        for (i, j), m in matches.items():
            matrix[i, j] = len(m)
            matrix[j, i] = len(m)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='YlOrRd')
        ax.set_xlabel('Image Index', fontsize=12)
        ax.set_ylabel('Image Index', fontsize=12)
        ax.set_title('Feature Matches Between Image Pairs', fontsize=14)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Number of Matches', fontsize=10)

        # Add text annotations for non-zero cells
        for i in range(n_images):
            for j in range(n_images):
                if matrix[i, j] > 0:
                    ax.text(j, i, int(matrix[i, j]),
                           ha='center', va='center', fontsize=8,
                           color='white' if matrix[i, j] > matrix.max()/2 else 'black')

        fig.tight_layout()

        # Convert to image
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    except ImportError:
        img = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.putText(img, "matplotlib required for plots", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img


def draw_feature_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    max_draw: int = 100,
) -> np.ndarray:
    """
    Draw feature matches between two images.

    Args:
        img1, img2: Input images
        kp1, kp2: Keypoints
        matches: Matches between keypoints
        max_draw: Maximum matches to draw

    Returns:
        Match visualization
    """
    # Sample if too many matches
    if len(matches) > max_draw:
        # Sort by distance and take best
        sorted_matches = sorted(matches, key=lambda m: m.distance)
        matches = sorted_matches[:max_draw]

    # Draw matches
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return vis


def draw_epipolar_lines(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    F: Optional[np.ndarray],
    max_lines: int = 20,
) -> np.ndarray:
    """
    Draw epipolar lines for matched features.

    Args:
        img1, img2: Input images
        kp1, kp2: Keypoints
        matches: Feature matches
        F: Fundamental matrix
        max_lines: Maximum lines to draw

    Returns:
        Epipolar visualization
    """
    if F is None or len(matches) == 0:
        # Fallback to simple match drawing
        return draw_feature_matches(img1, img2, kp1, kp2, matches, max_lines)

    # Sample matches
    if len(matches) > max_lines:
        indices = np.linspace(0, len(matches) - 1, max_lines, dtype=int)
        matches = [matches[i] for i in indices]

    vis1 = img1.copy()
    vis2 = img2.copy()
    h1, w1 = vis1.shape[:2]
    h2, w2 = vis2.shape[:2]

    colors = np.random.randint(0, 255, (len(matches), 3)).tolist()

    for i, m in enumerate(matches):
        pt1 = np.array([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], 1.0])
        pt2 = np.array([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1], 1.0])

        color = tuple(colors[i])

        # Epipolar line in image 2 from point in image 1
        l2 = F @ pt1
        x0, x1 = 0, w2
        if abs(l2[1]) > 1e-8:
            y0 = int(-l2[2] / l2[1])
            y1 = int(-(l2[2] + l2[0] * w2) / l2[1])
            cv2.line(vis2, (x0, y0), (x1, y1), color, 1)

        # Epipolar line in image 1 from point in image 2
        l1 = F.T @ pt2
        if abs(l1[1]) > 1e-8:
            y0 = int(-l1[2] / l1[1])
            y1 = int(-(l1[2] + l1[0] * w1) / l1[1])
            cv2.line(vis1, (0, y0), (w1, y1), color, 1)

        # Draw points
        cv2.circle(vis1, (int(pt1[0]), int(pt1[1])), 5, color, -1)
        cv2.circle(vis2, (int(pt2[0]), int(pt2[1])), 5, color, -1)

    # Stack images
    if h1 != h2:
        scale = h1 / h2
        vis2 = cv2.resize(vis2, (int(w2 * scale), h1))

    return np.hstack([vis1, vis2])


def debug_features(
    ctx: "DebugContext",
    images: list[np.ndarray],
    masks: list[np.ndarray],
    keypoints: list[list[cv2.KeyPoint]],
    matches: dict[tuple[int, int], list[cv2.DMatch]],
    fundamental_matrices: Optional[dict[tuple[int, int], np.ndarray]] = None,
) -> None:
    """
    Generate debug visualizations for the feature extraction stage.

    Outputs:
        - keypoints_XXX.jpg: Keypoints colored by scale
        - keypoint_heatmap_XXX.jpg: Density heatmaps
        - match_matrix.png: Match count heatmap
        - matches_I_J.jpg: Match visualizations for top pairs
        - epipolar_I_J.jpg: Epipolar line visualizations
        - match_stats.json: Feature statistics

    Args:
        ctx: Debug context
        images: Input images
        masks: Segmentation masks
        keypoints: Keypoints per image
        matches: Matches between pairs
        fundamental_matrices: F matrices per pair (optional)
    """
    if not ctx.enabled:
        return

    fundamental_matrices = fundamental_matrices or {}

    # Save keypoint visualizations
    for i, (img, kps) in enumerate(zip(images, keypoints)):
        # Keypoints colored by scale
        kp_vis = draw_keypoints_with_scale(img, kps)
        ctx.save_image("features", f"keypoints_{i:03d}", kp_vis)

        # Heatmap
        heatmap = create_keypoint_heatmap(img.shape[:2], kps)
        ctx.save_image("features", f"keypoint_heatmap_{i:03d}", heatmap)

    # Match matrix
    matrix_plot = create_match_matrix_plot(len(images), matches)
    ctx.save_image("features", "match_matrix.png", matrix_plot)

    # Top match pairs (by match count)
    sorted_pairs = sorted(matches.items(), key=lambda x: len(x[1]), reverse=True)
    for rank, ((i, j), pair_matches) in enumerate(sorted_pairs[:10]):
        # Match visualization
        match_vis = draw_feature_matches(
            images[i], images[j],
            keypoints[i], keypoints[j],
            pair_matches
        )
        ctx.save_image("features", f"matches_{i}_{j}", match_vis)

        # Epipolar lines
        F = fundamental_matrices.get((i, j))
        if F is not None:
            epipolar_vis = draw_epipolar_lines(
                images[i], images[j],
                keypoints[i], keypoints[j],
                pair_matches, F
            )
            ctx.save_image("features", f"epipolar_{i}_{j}", epipolar_vis)

    # Statistics
    total_matches = sum(len(m) for m in matches.values())
    stats = {
        "num_images": len(images),
        "keypoints_per_image": [len(kps) for kps in keypoints],
        "total_keypoints": sum(len(kps) for kps in keypoints),
        "num_image_pairs_matched": len(matches),
        "total_matches": total_matches,
        "matches_per_pair": {
            f"{i}_{j}": len(m) for (i, j), m in matches.items()
        },
        "top_pairs": [
            {"pair": f"{i}_{j}", "matches": len(m)}
            for (i, j), m in sorted_pairs[:10]
        ],
    }
    ctx.save_json("features", "match_stats", stats)
