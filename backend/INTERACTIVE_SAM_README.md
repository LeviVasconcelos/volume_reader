# Interactive SAM Segmentation Tool

A GUI tool for manually placing landmarks on images and running SAM segmentation.

## Usage

```bash
python interactive_sam.py <image_directory> [options]
```

### Examples

```bash
# Basic usage
python interactive_sam.py ./surfboard_frames/

# Save masks to specific directory
python interactive_sam.py ./surfboard_frames/ --output ./masks/

# Load previous annotations
python interactive_sam.py ./surfboard_frames/ --load annotations.json
```

## Controls

| Key | Action |
|-----|--------|
| `←` / `→` | Navigate between images |
| `p` | Switch to point mode |
| `b` | Switch to bbox mode |
| `c` | Clear current image annotations |
| `r` / `Enter` | Run SAM segmentation |
| `s` | Save annotations to JSON |
| `l` | Load annotations from JSON |
| `m` | Save masks to output directory |
| `q` / `Esc` | Quit |

## Feature Detection

| Key | Action |
|-----|--------|
| `f` | Detect keypoints in current image |
| `k` | Toggle keypoint visibility |
| `1` | Use SIFT detector (default) |
| `2` | Use ORB detector |
| `3` | Use AKAZE detector |

Keypoints are displayed as cyan dots. If a segmentation mask exists,
keypoints are only detected within the masked region.

## Feature Tracking

| Key | Action |
|-----|--------|
| `t` | Toggle tracking mode |

When tracking mode is enabled:
- Navigating between images automatically matches features between the previous and current image
- Matches are displayed in a separate "Feature Tracking" window (side-by-side view)
- Uses FLANN-based matching with geometric verification (RANSAC)
- Status shows match count (inliers/total)

**Note:** Tracking requires SIFT features (FLANN uses float descriptors).
ORB and AKAZE use binary descriptors which are not compatible with the FLANN matcher.
If you attempt tracking with non-SIFT features, a warning will be displayed.

### Tracking Workflow

1. Press `f` to detect SIFT features on image 1
2. Navigate to image 2 and press `f` to detect features
3. Press `t` to enable tracking mode
4. Navigate between images - tracking window will show feature matches
5. Press `t` again to disable tracking mode and close the tracking window

## Mouse Actions

**Point mode:**
- Left-click: Add foreground point
- Right-click: Remove nearest point (or clear bbox)

**Bbox mode:**
- Click + drag: Draw bounding box
- Right-click: Clear bbox (or remove nearest point)

## Workflow

1. Open images: `python interactive_sam.py ./my_images/`
2. Press `b` and draw a bounding box around the object
3. Press `p` and click to add foreground points on the object
4. Use arrow keys to annotate additional images
5. Press `r` to run SAM segmentation
6. Press `s` to save annotations, `m` to save masks
