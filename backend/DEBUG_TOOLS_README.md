# Debug Visualization Tools for Surfboard Volume Pipeline

This document explains how to use the debug visualization tools to diagnose and understand the 3D reconstruction pipeline.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Extracting Frames from Video](#extracting-frames-from-video)
- [Running the Debug Pipeline](#running-the-debug-pipeline)
- [Understanding the Output](#understanding-the-output)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

The debug tools generate visualizations for each of the 10 pipeline stages:

| Stage | Directory | What it shows |
|-------|-----------|---------------|
| 1. Preprocessing | `01_preprocess/` | Image gallery, camera intrinsics |
| 2. Segmentation | `02_segmentation/` | Mask overlays, click point, mask comparison |
| 3. Scale Detection | `03_scale/` | ArUco/card detection, scale consistency |
| 4. Feature Extraction | `04_features/` | Keypoints, match matrix, epipolar lines |
| 5. Structure from Motion | `05_sfm/` | Sparse point cloud, camera poses |
| 6. Dense Reconstruction | `06_dense/` | Depth maps, dense point cloud |
| 7. Mesh Generation | `07_mesh/` | Wireframe, shaded render, mesh quality |
| 8. Volume Calculation | `08_volume/` | Voxel slices, bounding box, dimensions |
| 9. Confidence Scoring | `09_confidence/` | Score breakdown, radar chart |
| 10. Object Verification | `10_object_verification/` | Match-based vs SAM mask comparison |

---

## Setup

### 1. Install Python Dependencies

```bash
cd /home/levi/workspace/volume_reader/backend
pip install -r requirements.txt
```

### 2. Download SAM Model Weights

The segmentation stage requires Facebook's Segment Anything Model (SAM):

```bash
# Create cache directory
mkdir -p ~/.cache/sam

# Option A: Download full model (2.4GB, best quality)
wget -O ~/.cache/sam/sam_vit_h.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Option B: Download smaller model (375MB, faster)
wget -O ~/.cache/sam/sam_vit_b.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 3. Verify Installation

```bash
python -c "import cv2, numpy, open3d, torch, matplotlib; print('All dependencies OK')"
```

---

## Quick Start

```bash
# 1. Extract frames from video
python extract_frames.py my_surfboard.mp4 ./frames/ --fps 1

# 2. Run debug pipeline with known dimensions
python run_debug_pipeline.py ./frames/ ./debug_output/ \
    --length 1830 --width 520 --thickness 63

# 3. View results
ls ./debug_output/
```

---

## Extracting Frames from Video

The pipeline requires a directory of images. Use `extract_frames.py` to extract frames from video:

### Basic Usage

```bash
python extract_frames.py <video_file> <output_directory> [options]
```

### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--fps F` | Extract F frames per second | `--fps 2` |
| `--every N` | Extract every Nth frame | `--every 30` |
| `--max M` | Maximum frames to extract | `--max 50` |

### Examples

```bash
# Extract 1 frame per second (default, good for most cases)
python extract_frames.py surfboard_video.mp4 ./frames/

# Extract 2 frames per second (more detail)
python extract_frames.py surfboard_video.mp4 ./frames/ --fps 2

# Extract every 15th frame from a 30fps video
python extract_frames.py surfboard_video.mp4 ./frames/ --every 15

# Extract maximum 40 frames (useful for testing)
python extract_frames.py surfboard_video.mp4 ./frames/ --max 40

# Combine options: 2 fps, max 60 frames
python extract_frames.py surfboard_video.mp4 ./frames/ --fps 2 --max 60
```

### Tips for Good Frame Extraction

- **Coverage**: Ensure the video shows the board from multiple angles (ideally 360°)
- **Frame count**: 20-40 frames is usually sufficient; more isn't always better
- **Overlap**: Adjacent frames should have ~60-80% visual overlap
- **Stability**: Avoid motion blur; steady footage works best

---

## Running the Debug Pipeline

### Basic Usage

```bash
python run_debug_pipeline.py <image_directory> <output_directory> [options]
```

### Scale Method Options

The pipeline needs to know the real-world scale. Three methods are available:

#### Method 1: User Dimensions (Recommended)

Provide known board dimensions in millimeters:

```bash
python run_debug_pipeline.py ./frames/ ./debug_output/ \
    --length 1830 \
    --width 520 \
    --thickness 63
```

You can provide 1, 2, or all 3 dimensions. More dimensions = better accuracy.

#### Method 2: ArUco Marker

If you have an ArUco marker in the scene:

```bash
python run_debug_pipeline.py ./frames/ ./debug_output/ \
    --scale-method aruco \
    --aruco-id 0 \
    --aruco-size 50
```

- `--aruco-id`: The marker ID (default: 0)
- `--aruco-size`: Physical marker size in mm

#### Method 3: Credit Card

If a credit card is visible in the scene:

```bash
python run_debug_pipeline.py ./frames/ ./debug_output/ \
    --scale-method credit_card
```

### Click Point Options

The segmentation uses a click point to identify the board. Default is center of first image.

```bash
# Use center of image 0 (default)
python run_debug_pipeline.py ./frames/ ./debug_output/ --length 1830

# Specify different image and position
python run_debug_pipeline.py ./frames/ ./debug_output/ \
    --length 1830 \
    --click-image 5 \
    --click-x 0.4 \
    --click-y 0.6
```

- `--click-image`: Image index (0-based)
- `--click-x`: Normalized X coordinate (0.0 = left, 1.0 = right)
- `--click-y`: Normalized Y coordinate (0.0 = top, 1.0 = bottom)

### Other Options

```bash
# Limit number of images processed
python run_debug_pipeline.py ./frames/ ./debug_output/ \
    --length 1830 \
    --max-images 20

# Skip certain stages (for faster debugging)
python run_debug_pipeline.py ./frames/ ./debug_output/ \
    --length 1830 \
    --skip-stages dense mesh
```

---

## Understanding the Output

### Directory Structure

```
debug_output/
├── 01_preprocess/
│   ├── image_gallery.jpg          # Grid of all input images
│   ├── intrinsics.json            # Camera parameters
│   └── preprocessing_summary.json
│
├── 02_segmentation/
│   ├── mask_overlay_000.jpg       # Each image with green mask overlay
│   ├── mask_overlay_001.jpg
│   ├── ...
│   ├── click_point.jpg            # Reference image with click point marked
│   ├── mask_comparison.jpg        # Grid comparing all masks
│   └── mask_stats.json            # Mask area %, connectivity stats
│
├── 03_scale/
│   ├── aruco_detection_000.jpg    # (if using ArUco)
│   ├── card_detection_000.jpg     # (if using credit card)
│   ├── scale_consistency.png      # Scale factor per image
│   └── scale_summary.json
│
├── 04_features/
│   ├── keypoints_000.jpg          # Keypoints colored by scale
│   ├── keypoint_heatmap_000.jpg   # Keypoint density heatmap
│   ├── match_matrix.png           # Heatmap of matches between pairs
│   ├── matches_0_1.jpg            # Feature matches visualization
│   ├── epipolar_0_1.jpg           # Epipolar line visualization
│   └── match_stats.json
│
├── 05_sfm/
│   ├── sparse_cloud.ply           # Sparse point cloud (open in MeshLab)
│   ├── sparse_cloud_top.png       # Top-down view
│   ├── sparse_cloud_side.png      # Side view
│   ├── camera_poses.png           # 3D plot of camera positions
│   ├── camera_poses.json          # Camera R, t matrices
│   ├── reprojection_errors.png    # Error histogram
│   └── triangulation_stats.json
│
├── 06_dense/
│   ├── depth_map_0_1.png          # Colored depth maps
│   ├── depth_validity_0_1.png     # Valid depth regions
│   ├── dense_cloud.ply            # Dense point cloud
│   ├── dense_cloud_view.png       # Rendered view
│   ├── normals_visualization.png  # Point cloud with normals
│   ├── outlier_removal.png        # Before/after outlier removal
│   └── dense_stats.json
│
├── 07_mesh/
│   ├── mesh.ply                   # Final mesh (open in MeshLab)
│   ├── mesh_wireframe.png         # Wireframe rendering
│   ├── mesh_shaded.png            # Shaded surface
│   ├── vertex_density.png         # Vertices colored by density
│   ├── mesh_issues.png            # Non-manifold edges highlighted
│   ├── mesh_holes.png             # Boundary edges (holes)
│   ├── smoothing_comparison.png   # Before/after smoothing
│   └── mesh_stats.json
│
├── 08_volume/
│   ├── voxel_grid.ply             # Voxelized representation
│   ├── voxel_slice_xy.png         # XY plane slice
│   ├── voxel_slice_xz.png         # XZ plane slice
│   ├── voxel_slice_yz.png         # YZ plane slice
│   ├── bounding_box.png           # Mesh with bounding boxes
│   ├── dimensions_annotated.png   # Dimension labels
│   └── volume_breakdown.json      # Detailed calculation
│
├── 09_confidence/
│   ├── score_breakdown.png        # Bar chart of component scores
│   ├── radar_chart.png            # Spider plot of all factors
│   ├── warnings_impact.png        # Warning penalties
│   └── confidence_summary.json
│
└── 10_object_verification/
    ├── projected_points_000.jpg   # 3D points projected to 2D
    ├── match_mask_000.jpg         # Segmentation from matches only
    ├── sam_mask_000.jpg           # SAM ground truth
    ├── mask_diff_000.jpg          # Difference (green=both, red=SAM, blue=match)
    ├── point_coverage_000.png     # Point density heatmap
    ├── iou_per_image.png          # IoU scores per image
    └── verification_metrics.json
```

### Viewing Results

#### PLY Files (Point Clouds and Meshes)

Open `.ply` files in:
- **MeshLab** (free): https://www.meshlab.net/
- **CloudCompare** (free): https://www.cloudcompare.org/
- **Blender** (free): https://www.blender.org/

#### PNG/JPG Images

Any image viewer, or:
```bash
# Linux
xdg-open debug_output/07_mesh/mesh_shaded.png

# macOS
open debug_output/07_mesh/mesh_shaded.png

# Or use Python
python -c "import cv2; cv2.imshow('img', cv2.imread('debug_output/07_mesh/mesh_shaded.png')); cv2.waitKey(0)"
```

#### JSON Files

```bash
# Pretty print JSON
python -m json.tool debug_output/08_volume/volume_breakdown.json

# Or use jq
jq . debug_output/08_volume/volume_breakdown.json
```

---

## Examples

### Example 1: Basic Run with Known Dimensions

```bash
# Extract frames
python extract_frames.py surfboard_360.mp4 ./frames/ --fps 1

# Run pipeline
python run_debug_pipeline.py ./frames/ ./debug_output/ \
    --length 1830 --width 520 --thickness 63

# Check results
cat debug_output/08_volume/volume_breakdown.json | python -m json.tool
```

### Example 2: Debugging Segmentation Issues

If segmentation looks wrong, adjust the click point:

```bash
# First, check which image shows the board clearly
ls ./frames/

# Run with specific click point
python run_debug_pipeline.py ./frames/ ./debug_output/ \
    --length 1830 \
    --click-image 10 \
    --click-x 0.5 \
    --click-y 0.4

# Check segmentation results
xdg-open debug_output/02_segmentation/mask_comparison.jpg
```

### Example 3: Quick Test with Limited Frames

```bash
# Extract only 15 frames for quick testing
python extract_frames.py surfboard.mp4 ./frames/ --max 15

# Run pipeline
python run_debug_pipeline.py ./frames/ ./debug_output/ --length 1830

# Check if reconstruction worked
xdg-open debug_output/05_sfm/camera_poses.png
```

### Example 4: Using ArUco Marker for Scale

```bash
# If you have a 50mm ArUco marker (ID 0) in the scene
python run_debug_pipeline.py ./frames/ ./debug_output/ \
    --scale-method aruco \
    --aruco-id 0 \
    --aruco-size 50

# Check marker detection
xdg-open debug_output/03_scale/aruco_detection_000.jpg
```

### Example 5: Analyzing Feature Matching

```bash
# Run pipeline
python run_debug_pipeline.py ./frames/ ./debug_output/ --length 1830

# Check feature matching quality
xdg-open debug_output/04_features/match_matrix.png

# If matches look sparse, check individual pairs
xdg-open debug_output/04_features/matches_0_1.jpg
```

### Example 6: Investigating Low Confidence

```bash
# Run pipeline
python run_debug_pipeline.py ./frames/ ./debug_output/ --length 1830

# Check what's causing low confidence
xdg-open debug_output/09_confidence/score_breakdown.png
xdg-open debug_output/09_confidence/radar_chart.png

# Read warnings
cat debug_output/09_confidence/confidence_summary.json | python -m json.tool
```

---

## Troubleshooting

### Problem: "SAM model checkpoint not found"

**Solution**: Download the SAM weights:
```bash
mkdir -p ~/.cache/sam
wget -O ~/.cache/sam/sam_vit_h.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Problem: Segmentation masks are wrong

**Solutions**:
1. Try a different click point:
   ```bash
   python run_debug_pipeline.py ./frames/ ./debug_output/ \
       --length 1830 --click-image 5 --click-x 0.5 --click-y 0.5
   ```
2. Check `02_segmentation/click_point.jpg` to see where the click landed
3. Ensure the board is visible and not occluded at the click point

### Problem: Few feature matches

**Possible causes**:
- Images have motion blur
- Not enough overlap between frames
- Board surface is too uniform (no texture)

**Solutions**:
- Extract more frames: `--fps 2`
- Ensure video has good lighting and focus
- Check `04_features/match_matrix.png` to see which pairs match well

### Problem: Sparse point cloud is empty or wrong

**Check**:
1. `05_sfm/camera_poses.png` - Are cameras positioned sensibly?
2. `04_features/match_matrix.png` - Are there enough matches?
3. Try with more images or better frame selection

### Problem: Mesh has holes

**This is normal for single-view reconstructions. Check**:
- `07_mesh/mesh_holes.png` - Shows boundary edges
- `07_mesh/mesh_stats.json` - `is_watertight` field

### Problem: Volume seems wrong

**Check**:
1. `08_volume/dimensions_annotated.png` - Are dimensions reasonable?
2. `08_volume/volume_breakdown.json` - Check scale factor
3. Verify your input dimensions are correct (in mm)

### Problem: Out of memory

**Solutions**:
- Reduce image count: `--max-images 20`
- Extract fewer frames: `--fps 0.5`
- Use smaller images (resize before processing)

---

## API Integration

To use debug output from the API, pass `debug_dir` to `run_pipeline()`:

```python
from pathlib import Path
from app.pipeline import run_pipeline

# In your job handler:
await run_pipeline(job, debug_dir=Path("./debug_output"))
```

---

## File Descriptions

### Key Files to Check First

| File | What to look for |
|------|------------------|
| `02_segmentation/mask_comparison.jpg` | Are all masks capturing the board correctly? |
| `04_features/match_matrix.png` | Do most image pairs have matches (yellow/red cells)? |
| `05_sfm/camera_poses.png` | Are cameras arranged in a sensible pattern? |
| `07_mesh/mesh_shaded.png` | Does the mesh look like a surfboard? |
| `09_confidence/score_breakdown.png` | Which components are scoring low? |

### JSON Files Reference

| File | Key fields |
|------|------------|
| `preprocessing_summary.json` | `num_images`, `processed_dimensions` |
| `mask_stats.json` | `area_percent`, `num_components` per image |
| `scale_summary.json` | `final_scale_factor`, `method` |
| `match_stats.json` | `total_matches`, `matches_per_pair` |
| `triangulation_stats.json` | `num_3d_points`, `num_cameras` |
| `dense_stats.json` | `num_points`, point cloud info |
| `mesh_stats.json` | `num_vertices`, `num_triangles`, `is_watertight` |
| `volume_breakdown.json` | `volume_liters`, `dimensions`, `scale_factor` |
| `confidence_summary.json` | `final_score`, `component_scores`, `warnings` |
| `verification_metrics.json` | `mean_iou`, `low_iou_images` |
