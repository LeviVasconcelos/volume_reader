# Debug Visualization Tools Plan

## Overview
Create debug visualization tools for each pipeline stage that output images/artifacts to disk for debugging reconstruction issues.

## Architecture

### File Structure
```
backend/
├── app/
│   └── pipeline/
│       └── debug/
│           ├── __init__.py                   # Debug context manager & utilities
│           ├── preprocess_debug.py           # Stage 1: Image preprocessing
│           ├── segmentation_debug.py         # Stage 2: SAM segmentation
│           ├── scale_debug.py                # Stage 3: Scale detection
│           ├── features_debug.py             # Stage 4: Feature extraction & matching
│           ├── sfm_debug.py                  # Stage 5: Structure from Motion
│           ├── dense_debug.py                # Stage 6: Dense reconstruction
│           ├── mesh_debug.py                 # Stage 7: Mesh generation
│           ├── volume_debug.py               # Stage 8: Volume calculation
│           ├── confidence_debug.py           # Stage 9: Confidence scoring
│           └── object_verification_debug.py  # Stage 10: Match vs SAM verification
├── run_debug_pipeline.py                     # CLI debug runner
```

### Core Debug Infrastructure

**File: `debug/__init__.py`**
```python
class DebugContext:
    """Manages debug output directory and settings."""
    def __init__(self, output_dir: Path, enabled: bool = True):
        self.output_dir = output_dir
        self.enabled = enabled
        self.stage_dirs = {}

    def stage_dir(self, stage_name: str) -> Path:
        """Get/create directory for a pipeline stage."""

    def save_image(self, stage: str, name: str, image: np.ndarray) -> Path:
        """Save debug image with consistent naming."""

    def save_json(self, stage: str, name: str, data: dict) -> Path:
        """Save debug metadata as JSON."""

    def save_point_cloud(self, stage: str, name: str, cloud) -> Path:
        """Save point cloud as PLY file."""

    def save_mesh(self, stage: str, name: str, mesh) -> Path:
        """Save mesh as PLY/OBJ file."""
```

---

## Stage-by-Stage Debug Visualizations

### Stage 1: Preprocessing (`preprocess_debug.py`)

**Outputs:**
- `01_preprocess/`
  - `image_gallery.jpg` - Grid of all input images with dimensions labeled
  - `intrinsics.json` - Camera intrinsics (fx, fy, cx, cy, source: EXIF/estimated)
  - `preprocessing_summary.json` - Original vs processed dimensions per image

**Functions:**
```python
def debug_preprocessing(
    ctx: DebugContext,
    images: list[np.ndarray],
    intrinsics: CameraIntrinsics,
    image_paths: list[Path]
) -> None
```

---

### Stage 2: Segmentation (`segmentation_debug.py`)

**Outputs:**
- `02_segmentation/`
  - `mask_overlay_{i:03d}.jpg` - Each image with mask overlay (semi-transparent green)
  - `click_point.jpg` - Reference image with click point marked
  - `mask_comparison.jpg` - Grid showing all masks side by side
  - `mask_stats.json` - Per-image: mask area %, connectivity, bounding box

**Functions:**
```python
def debug_segmentation(
    ctx: DebugContext,
    images: list[np.ndarray],
    masks: list[np.ndarray],
    click_point: tuple[float, float],
    reference_idx: int
) -> None
```

---

### Stage 3: Scale Detection (`scale_debug.py`)

**Outputs:**
- `03_scale/`
  - `aruco_detection_{i:03d}.jpg` - Images with detected ArUco markers boxed
  - `card_detection_{i:03d}.jpg` - Images with detected credit card corners
  - `scale_consistency.png` - Plot of scale factor per image with median line
  - `scale_summary.json` - Method used, scale factor, confidence, detections per image

**Functions:**
```python
def debug_scale_detection(
    ctx: DebugContext,
    images: list[np.ndarray],
    scale_factor: float,
    method: str,
    detections: list[dict]  # Per-image detection results
) -> None
```

---

### Stage 4: Feature Extraction & Matching (`features_debug.py`)

**Outputs:**
- `04_features/`
  - `keypoints_{i:03d}.jpg` - Each image with SIFT keypoints drawn (colored by scale)
  - `keypoint_heatmap_{i:03d}.jpg` - Density heatmap of keypoint locations
  - `match_matrix.png` - Heatmap showing match counts between all image pairs
  - `matches_{i}_{j}.jpg` - Top image pairs with matched features drawn
  - `epipolar_{i}_{j}.jpg` - Feature matches with epipolar lines
  - `match_stats.json` - Total keypoints, matches per pair, inlier ratios

**Functions:**
```python
def debug_features(
    ctx: DebugContext,
    images: list[np.ndarray],
    masks: list[np.ndarray],
    keypoints: list[list[cv2.KeyPoint]],
    matches: dict[tuple[int,int], list[cv2.DMatch]],
    fundamental_matrices: dict[tuple[int,int], np.ndarray]
) -> None
```

---

### Stage 5: Structure from Motion (`sfm_debug.py`)

**Outputs:**
- `05_sfm/`
  - `sparse_cloud.ply` - Sparse point cloud (viewable in MeshLab/CloudCompare)
  - `sparse_cloud_top.png` - Top-down view of sparse cloud
  - `sparse_cloud_side.png` - Side view of sparse cloud
  - `camera_poses.png` - 3D plot with camera frustums and optical axes
  - `camera_poses.json` - R, t matrices for each camera
  - `reprojection_errors.png` - Histogram of reprojection errors
  - `triangulation_stats.json` - Points per image pair, outliers removed
  - `bundle_adjustment.png` - Before/after error comparison

**Functions:**
```python
def debug_sfm(
    ctx: DebugContext,
    sparse_cloud: o3d.geometry.PointCloud,
    camera_poses: list[CameraPose],
    reprojection_errors: list[float],
    triangulation_stats: dict
) -> None
```

---

### Stage 6: Dense Reconstruction (`dense_debug.py`)

**Outputs:**
- `06_dense/`
  - `depth_map_{i}_{j}.png` - Pseudocolor depth maps for each stereo pair
  - `depth_validity_{i}_{j}.png` - Binary mask of valid depth regions
  - `rectified_{i}_{j}.jpg` - Stereo rectified image pairs
  - `dense_cloud.ply` - Full dense point cloud
  - `dense_cloud_view.png` - Rendered view of dense cloud
  - `outlier_removal.png` - Before/after statistical outlier removal
  - `normals_visualization.png` - Point cloud with normal vectors as arrows
  - `dense_stats.json` - Points per pair, merge stats, outlier counts

**Functions:**
```python
def debug_dense(
    ctx: DebugContext,
    depth_maps: list[tuple[int, int, np.ndarray]],  # (i, j, depth)
    dense_cloud: o3d.geometry.PointCloud,
    intermediate_clouds: list[o3d.geometry.PointCloud],
    outlier_stats: dict
) -> None
```

---

### Stage 7: Mesh Generation (`mesh_debug.py`)

**Outputs:**
- `07_mesh/`
  - `mesh.ply` - Final mesh (viewable in MeshLab)
  - `mesh_wireframe.png` - Wireframe rendering
  - `mesh_shaded.png` - Shaded surface rendering
  - `vertex_density.png` - Vertices colored by Poisson density
  - `mesh_issues.png` - Highlighted degenerate/non-manifold areas
  - `mesh_holes.png` - Visualization of holes (if not watertight)
  - `smoothing_comparison.png` - Before/after Laplacian smoothing
  - `mesh_stats.json` - Vertex/face count, watertight status, method used

**Functions:**
```python
def debug_mesh(
    ctx: DebugContext,
    mesh: o3d.geometry.TriangleMesh,
    densities: np.ndarray,
    is_watertight: bool,
    method_used: str,
    pre_smooth_mesh: o3d.geometry.TriangleMesh
) -> None
```

---

### Stage 8: Volume Calculation (`volume_debug.py`)

**Outputs:**
- `08_volume/`
  - `voxel_grid.ply` - Voxelized mesh
  - `voxel_slice_xy.png` - XY plane slice through voxel grid
  - `voxel_slice_xz.png` - XZ plane slice
  - `voxel_slice_yz.png` - YZ plane slice
  - `bounding_box.png` - Mesh with oriented bounding box overlay
  - `dimensions_annotated.png` - Mesh with length/width/thickness labels
  - `volume_breakdown.json` - Voxel size, count, raw volume, scaled volume

**Functions:**
```python
def debug_volume(
    ctx: DebugContext,
    mesh: o3d.geometry.TriangleMesh,
    voxel_grid: o3d.geometry.VoxelGrid,
    voxel_size: float,
    dimensions: dict,
    scale_factor: float
) -> None
```

---

### Stage 9: Confidence Scoring (`confidence_debug.py`)

**Outputs:**
- `09_confidence/`
  - `score_breakdown.png` - Bar chart of all 6 component scores with weights
  - `radar_chart.png` - Hexagon/radar plot of all factors
  - `confidence_summary.json` - All component scores, weights, final score
  - `warnings_impact.png` - Which warnings caused score penalties

**Functions:**
```python
def debug_confidence(
    ctx: DebugContext,
    component_scores: dict[str, float],
    weights: dict[str, float],
    final_score: float,
    warnings: list[str]
) -> None
```

---

### Stage 10: Match-Based Segmentation Verification (`object_verification_debug.py`)

**Purpose:** Verify correct 3D object identification by segmenting board pixels using only dense feature matches, then comparing against SAM segmentation as ground truth.

**Concept:**
- Project triangulated 3D points back to each image
- Grow regions from matched feature locations (flood-fill or superpixel)
- Build a "match-derived" segmentation mask without using SAM
- Compare against SAM masks to validate reconstruction is tracking the correct object

**Algorithm:**
```
1. Get all 3D points from SfM/dense reconstruction
2. For each image:
   a. Project 3D points to 2D using camera pose
   b. Filter to points within image bounds
   c. Create seed points from valid projections
   d. Region growing from seeds:
      - Use color similarity to expand regions
      - Or use SLIC superpixels and merge those containing seeds
   e. Generate "match-based mask"
3. Compare match-based mask vs SAM mask:
   - IoU (Intersection over Union)
   - Precision/Recall
   - Boundary accuracy
```

**Outputs:**
- `10_object_verification/`
  - `projected_points_{i:03d}.jpg` - Images with 3D→2D projected points overlaid
  - `match_mask_{i:03d}.jpg` - Segmentation derived from matches only
  - `sam_mask_{i:03d}.jpg` - SAM ground truth mask
  - `mask_comparison_{i:03d}.jpg` - Side-by-side or overlay comparison
  - `mask_diff_{i:03d}.jpg` - Difference visualization (red=SAM only, blue=match only, green=both)
  - `iou_per_image.png` - Bar chart of IoU scores per image
  - `verification_metrics.json` - Per-image and aggregate metrics:
    - IoU, precision, recall per image
    - Mean IoU across all images
    - Outlier images (low IoU = potential tracking failure)
  - `point_coverage.png` - Heatmap of projected point density per image

**Functions:**
```python
def debug_object_verification(
    ctx: DebugContext,
    images: list[np.ndarray],
    sam_masks: list[np.ndarray],
    sparse_cloud: o3d.geometry.PointCloud,
    dense_cloud: o3d.geometry.PointCloud,
    camera_poses: list[CameraPose],
    intrinsics: CameraIntrinsics
) -> dict:
    """
    Verify object identification by comparing match-based vs SAM segmentation.

    Returns:
        dict with per-image IoU, precision, recall and aggregate metrics
    """

def project_points_to_image(
    points_3d: np.ndarray,
    camera_pose: CameraPose,
    intrinsics: CameraIntrinsics,
    image_shape: tuple[int, int]
) -> np.ndarray:
    """Project 3D points to 2D image coordinates."""

def grow_mask_from_seeds(
    image: np.ndarray,
    seed_points: np.ndarray,
    method: str = "superpixel"  # or "flood_fill" or "grabcut"
) -> np.ndarray:
    """
    Generate segmentation mask by growing from seed points.

    Methods:
    - superpixel: Use SLIC, merge superpixels containing seeds
    - flood_fill: Color-based region growing from each seed
    - grabcut: Use seeds as foreground hints for GrabCut
    """

def compute_mask_metrics(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray
) -> dict:
    """Compute IoU, precision, recall, boundary accuracy."""
```

**Use Cases:**
1. **Detect tracking failures**: If match-based mask has low IoU with SAM, the feature matches may be tracking the wrong object or background
2. **Validate turntable detection**: In turntable mode, background features should NOT project to board region
3. **Identify problematic views**: Images with low IoU may have poor feature coverage or occlusions
4. **Debug multi-object scenes**: Helps verify reconstruction is focused on the correct object

---

## Integration

### Pipeline Integration
Modify `pipeline/__init__.py` to accept debug context:

```python
async def run_pipeline(
    job: Job,
    image_paths: list[Path],
    request: JobRequest,
    debug_dir: Optional[Path] = None  # NEW: Enable debug output
) -> PipelineResult:

    debug = DebugContext(debug_dir) if debug_dir else None

    # After each stage, call debug function if enabled
    images, intrinsics = await preprocess_images(image_paths)
    if debug:
        debug_preprocessing(debug, images, intrinsics, image_paths)

    # ... continue for each stage
```

### CLI Debug Runner
Create standalone debug script for testing:

**File: `backend/run_debug_pipeline.py`**
```bash
python run_debug_pipeline.py ./test_images/ ./debug_output/ \
  --scale-method user_dimensions \
  --length 1830 --width 520 --thickness 63
```

Runs full pipeline with all debug visualizations enabled.

---

## Files to Create
1. `backend/app/pipeline/debug/__init__.py` - DebugContext class
2. `backend/app/pipeline/debug/preprocess_debug.py`
3. `backend/app/pipeline/debug/segmentation_debug.py`
4. `backend/app/pipeline/debug/scale_debug.py`
5. `backend/app/pipeline/debug/features_debug.py`
6. `backend/app/pipeline/debug/sfm_debug.py`
7. `backend/app/pipeline/debug/dense_debug.py`
8. `backend/app/pipeline/debug/mesh_debug.py`
9. `backend/app/pipeline/debug/volume_debug.py`
10. `backend/app/pipeline/debug/confidence_debug.py`
11. `backend/app/pipeline/debug/object_verification_debug.py` - Match-based segmentation vs SAM
12. `backend/run_debug_pipeline.py` - CLI debug runner

## Files to Modify
1. `backend/app/pipeline/__init__.py` - Add optional debug_dir parameter

## Dependencies
- matplotlib (for plots/charts) - ADD to requirements.txt
- No new major dependencies (uses existing opencv, open3d, numpy)

## Verification
1. Run debug pipeline on test images
2. Verify all 9 stage directories created with expected outputs
3. Open PLY files in MeshLab/CloudCompare
4. Review PNG visualizations for each stage
5. Check JSON metadata files are valid and informative
