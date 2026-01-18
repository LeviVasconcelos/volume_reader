# Backend Pipeline Plan

## Overview
Monolithic FastAPI service for processing surfboard images and computing volume via photogrammetry.

## API Design

### Endpoints
```
POST /jobs                  - Create new reconstruction job
GET  /jobs/{job_id}         - Poll job status and results
GET  /jobs/{job_id}/status  - Lightweight status check
DELETE /jobs/{job_id}       - Cancel/cleanup job
```

### Job Flow
1. Client uploads images + metadata (scale method, user click point)
2. Server returns job_id immediately
3. Client polls until status is "completed" or "failed"
4. Results include: volume, dimensions, confidence, warnings

### Request Payload
```json
{
  "images": ["base64 or multipart upload"],
  "scale_method": "aruco" | "credit_card" | "user_dimensions",
  "scale_data": {
    "aruco_id": 42,
    "aruco_size_mm": 50
  } | {
    "card_type": "standard"
  } | {
    "length_mm": 1830,
    "width_mm": 520,
    "thickness_mm": 63
  },
  "board_click_point": {
    "image_index": 0,
    "x": 0.5,
    "y": 0.5
  }
}
```

### Response (completed job)
```json
{
  "job_id": "uuid",
  "status": "completed",
  "result": {
    "volume_liters": 32.5,
    "dimensions": {
      "length_mm": 1829,
      "width_mm": 518,
      "thickness_mm": 62
    },
    "confidence": 0.92,
    "warnings": ["Minor gaps in tail reconstruction"]
  }
}
```

## Pipeline Stages

### 0. Video to Images (Optional Pre-processing)
For video input, use `video_to_images.py` to extract optimal frames:
- **Motion-based extraction**: Select frames when sufficient visual change occurs
- **Blur detection**: Reject frames below sharpness threshold (Laplacian variance)
- **Capture mode detection**:
  - Camera-moving: Standard photogrammetry workflow
  - Turntable: Static background detected, flag for stricter masking
- **Output**: Sequential frames + metadata JSON with timestamps, scores, and mode

```bash
python video_to_images.py input.mp4 output_dir/
  --min-motion 0.02      # Minimum motion threshold (0-1)
  --min-sharpness 100    # Minimum Laplacian variance
  --max-frames 50        # Maximum frames to extract
  --format jpg|png       # Output format
```

### 1. Image Preprocessing
- Decode uploaded images (or frames from video extraction)
- Extract EXIF for camera intrinsics (if available)
- Store in temp working directory

### 2. Segmentation (SAM-based)
- Load SAM model (GPU accelerated)
- Use user click point as prompt
- Segment surfboard from background in all images
- Generate masks for reconstruction

### 3. Scale Reference Detection
Supports three methods:

**ArUco Markers:**
- OpenCV ArUco detection
- Extract marker corners, compute real-world scale

**Credit Card Detection:**
- Detect rectangular card shape (85.6mm x 53.98mm standard)
- Use edge detection + contour analysis
- Validate aspect ratio matches credit card

**User Dimensions:**
- Skip detection
- Apply scale after reconstruction based on provided length/width/thickness

### 4. Feature Extraction & Matching (SIFT)
- Extract SIFT features from masked regions
- Match features across image pairs
- Filter matches with ratio test
- Build feature tracks

### 5. Structure from Motion
- Estimate camera intrinsics (or use EXIF)
- Incremental SfM:
  - Initialize with best image pair
  - Triangulate initial points
  - Add images incrementally via PnP
  - Bundle adjustment after each addition
- Output: sparse point cloud + camera poses

### 6. Dense Reconstruction
- Multi-view stereo on masked regions
- Generate dense point cloud
- Statistical outlier removal

### 7. Mesh Generation
- Poisson surface reconstruction or Ball-pivoting
- Mesh smoothing
- Hole filling for minor gaps

### 8. Scale Calibration
- Apply scale factor from reference detection
- Or fit reconstructed model to user-provided dimensions
- Cross-validate if multiple scale sources available

### 9. Volume Calculation (Voxelization)
- Adaptive voxel resolution: ~1-2mm based on board size
- Voxelize mesh interior
- Count occupied voxels
- Convert to liters: `volume_mm³ / 1,000,000`

### 10. Dimension Extraction
- Compute oriented bounding box
- Extract length, width, thickness
- Compare with user input if provided (validation)

### 11. Confidence Scoring
Factors affecting confidence:
- Number of successfully matched images
- Point cloud density / coverage
- Mesh quality (holes, noise)
- Scale calibration certainty
- Dimension consistency (if user provided)

### 12. Result Assembly
- Compile volume, dimensions, confidence
- Generate warnings for specific issues:
  - "Gaps detected in nose/tail area"
  - "Low feature coverage on bottom"
  - "Scale reference partially occluded"
- Cleanup intermediate files (keep 24-48h for debugging)

## Tech Stack

### Core Libraries
- **FastAPI**: API framework
- **OpenCV**: Image processing, SIFT, ArUco detection
- **Open3D**: Point cloud processing, mesh operations, voxelization
- **Segment Anything (SAM)**: Board segmentation
- **NumPy/SciPy**: Numerical operations
- **Pillow**: Image I/O

### Infrastructure
- **Docker + docker-compose**: Containerization
- **Redis** (optional): Job queue if needed later
- **GPU**: Required for SAM, beneficial for dense reconstruction

## Directory Structure
```
backend/
├── app/
│   ├── main.py              # FastAPI app, routes
│   ├── models.py            # Pydantic models
│   ├── jobs.py              # Job management
│   └── pipeline/
│       ├── __init__.py
│       ├── preprocess.py    # Image preprocessing
│       ├── segmentation.py  # SAM segmentation
│       ├── scale.py         # Scale detection (ArUco, card, user)
│       ├── features.py      # SIFT extraction & matching
│       ├── sfm.py           # Structure from Motion
│       ├── dense.py         # Dense reconstruction
│       ├── mesh.py          # Mesh generation & repair
│       ├── volume.py        # Voxelization & volume calc
│       └── confidence.py    # Confidence scoring
├── tests/
├── video_to_images.py       # Video frame extraction for photogrammetry
├── client.py                # Test client script
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Data Retention
- Jobs and images retained for 24-48 hours
- Background cleanup task removes expired data
- Allows debugging and retry without re-upload

## Error Handling (Tiered Response)
Always return a response with:
- `status`: "completed", "failed", "partial"
- `confidence`: 0.0 - 1.0
- `warnings`: List of specific issues
- `errors`: List of fatal problems (if failed)

Examples:
- High confidence (>0.85): Clean result, no warnings
- Medium confidence (0.6-0.85): Result with warnings about gaps/uncertainty
- Low confidence (<0.6): Result marked as "partial", strong warnings
- Failed: No volume, list of errors explaining why
