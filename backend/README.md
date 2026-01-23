# Surfboard Volume Calculator - Backend

A FastAPI-based backend service that computes surfboard volumes from multi-view images using photogrammetry and 3D reconstruction.

## Overview

This service accepts multiple images of a surfboard, reconstructs a 3D model using Structure from Motion (SfM), and calculates the volume with a target accuracy of ±0.5 liters.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                          │
├─────────────────────────────────────────────────────────────────┤
│  POST /jobs     - Upload images, create reconstruction job      │
│  GET /jobs/{id} - Poll for results                              │
│  DELETE /jobs   - Cancel/cleanup job                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Processing Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Preprocess    - Load images, extract camera intrinsics      │
│  2. Segmentation  - SAM-based board segmentation (user click)   │
│  3. Scale Detect  - ArUco / Credit card / User dimensions       │
│  4. Features      - SIFT extraction & matching                  │
│  5. SfM           - Structure from Motion → sparse cloud        │
│  6. Dense         - Multi-view stereo → dense cloud             │
│  7. Mesh          - Poisson surface reconstruction              │
│  8. Volume        - Voxelization & volume calculation           │
│  9. Confidence    - Quality scoring                             │
└─────────────────────────────────────────────────────────────────┘
```

## API Endpoints

### `POST /jobs`
Create a new volume calculation job.

**Request:** `multipart/form-data`
- `images`: Multiple image files (minimum 3)
- `request_data`: JSON string with job configuration

```json
{
  "scale_method": "user_dimensions",
  "user_dimensions": {
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

**Scale methods:**
- `aruco` - Detect ArUco marker (requires `aruco_data`)
- `credit_card` - Detect credit card for scale
- `user_dimensions` - Use provided dimensions (requires `user_dimensions`)

**Response:**
```json
{
  "job_id": "uuid",
  "status": "pending",
  "progress": "Job created, queued for processing"
}
```

### `GET /jobs/{job_id}`
Poll job status and results.

**Response (completed):**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "progress": "Completed",
  "result": {
    "volume_liters": 32.5,
    "dimensions": {
      "length_mm": 1829,
      "width_mm": 518,
      "thickness_mm": 62
    },
    "confidence": 0.92,
    "warnings": []
  },
  "errors": []
}
```

**Status values:** `pending`, `processing`, `completed`, `partial`, `failed`

### `GET /jobs/{job_id}/status`
Lightweight status check (no result data).

### `DELETE /jobs/{job_id}`
Cancel and cleanup a job.

### `GET /health`
Health check endpoint.

## Setup

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- NVIDIA GPU with CUDA (recommended for SAM segmentation)

### Local Development

1. **Create virtual environment:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download SAM model:**
```bash
mkdir -p models
# Download ViT-H (recommended, 2.4GB):
curl -L -o models/sam_vit_h.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Or ViT-B (smaller, 375MB):
curl -L -o models/sam_vit_b.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

4. **Run the server:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

1. **With GPU support:**
```bash
docker-compose up --build
```

2. **CPU only (for testing):**
```bash
docker-compose --profile cpu up --build api-cpu
```

The API will be available at `http://localhost:8000`.

## Testing

### Unit Tests

Run the test suite:
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_scale.py -v

# With coverage
pytest tests/ -v --cov=app --cov-report=html
```

Test files:
- `test_preprocess.py` - Image loading, camera intrinsics
- `test_scale.py` - Scale detection (ArUco, credit card, user input)
- `test_features.py` - SIFT extraction and matching
- `test_volume.py` - Voxelization and volume calculation
- `test_confidence.py` - Confidence scoring
- `test_api.py` - API endpoint tests

### Integration Testing with Client Script

The `tools/client.py` script provides an easy way to test the full pipeline:

```bash
# Using user-provided dimensions
python tools/client.py ./surfboard_photos \
  --length 1830 \
  --width 520 \
  --thickness 63

# Using ArUco marker
python tools/client.py ./surfboard_photos \
  --scale-method aruco \
  --aruco-id 42 \
  --aruco-size 50

# Using credit card
python tools/client.py ./surfboard_photos \
  --scale-method credit_card

# Custom server URL
python tools/client.py ./surfboard_photos \
  --url http://your-server:8000 \
  --length 1830
```

**Client options:**
```
positional arguments:
  image_dir             Directory containing surfboard images

options:
  --url URL             API base URL (default: http://localhost:8000)
  --scale-method        Scale method: user_dimensions, aruco, credit_card
  --length MM           Board length in mm
  --width MM            Board width in mm
  --thickness MM        Board thickness in mm
  --aruco-id ID         ArUco marker ID
  --aruco-size MM       ArUco marker size in mm
  --click-image INDEX   Image index for board click point (default: 0)
  --click-x X           Normalized X coordinate for click (default: 0.5)
  --click-y Y           Normalized Y coordinate for click (default: 0.5)
  --poll-interval SEC   Polling interval in seconds (default: 2)
```

### Manual API Testing

Using `curl`:

```bash
# Create job
curl -X POST http://localhost:8000/jobs \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg" \
  -F "images=@photo3.jpg" \
  -F 'request_data={
    "scale_method": "user_dimensions",
    "user_dimensions": {"length_mm": 1830, "width_mm": 520, "thickness_mm": 63},
    "board_click_point": {"image_index": 0, "x": 0.5, "y": 0.5}
  }'

# Poll for results
curl http://localhost:8000/jobs/{job_id}

# Check health
curl http://localhost:8000/health
```

Using the interactive docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Video to Images Conversion

The `tools/video_to_images.py` script extracts optimal frames from video for photogrammetry reconstruction. It supports both camera-moving and turntable (board-rotating) capture scenarios.

### Basic Usage

```bash
# Extract frames from video
python tools/video_to_images.py surfboard_video.mp4 ./frames/

# Extract more frames with lower motion threshold
python tools/video_to_images.py video.mp4 ./output/ --min-motion 0.01 --max-frames 80

# Higher quality PNG output
python tools/video_to_images.py video.mp4 ./output/ --format png --min-sharpness 150
```

### Options

```
positional arguments:
  input                 Input video file path
  output                Output directory for extracted frames

options:
  --min-motion FLOAT    Minimum motion threshold (0-1) between frames (default: 0.02)
  --min-sharpness FLOAT Minimum Laplacian variance for sharpness (default: 100)
  --max-frames INT      Maximum number of frames to extract (default: 50)
  --format {jpg,png}    Output image format (default: jpg)
  --quiet               Suppress progress output
```

### Features

- **Motion-based extraction**: Frames are selected when sufficient visual change occurs, avoiding redundant or nearly identical frames
- **Blur detection**: Blurry frames are automatically rejected using Laplacian variance analysis
- **Turntable detection**: Automatically detects if the video shows a rotating object with static background and flags it for the pipeline to use stricter masking
- **Metadata output**: Generates `extraction_metadata.json` with frame timestamps, motion scores, blur scores, and capture mode detection

### Output

The script creates:
- Sequentially named frame images (`frame_0000.jpg`, `frame_0001.jpg`, etc.)
- `extraction_metadata.json` containing:
  - Detected capture mode (`camera_moving` or `turntable`)
  - Turntable confidence score
  - Per-frame metadata (timestamps, motion/blur scores)
  - Any warnings about extraction quality

### Workflow

1. Record video walking around the surfboard (camera-moving) or rotating the board on a turntable
2. Run `tools/video_to_images.py` to extract frames
3. Use extracted frames with the volume calculation pipeline via `tools/client.py`

```bash
# Full workflow example
python tools/video_to_images.py surfboard.mp4 ./surfboard_frames/
python tools/client.py ./surfboard_frames --length 1830 --width 520 --thickness 63
```

## Project Structure

```
backend/
├── README.md                 # Main documentation
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore                # Ignores debug outputs, data/, venv/
│
├── app/                      # Core application
│   ├── __init__.py
│   ├── main.py              # FastAPI app and routes
│   ├── models.py            # Pydantic request/response models
│   ├── jobs.py              # Job management
│   └── pipeline/
│       ├── __init__.py      # Pipeline orchestrator
│       ├── preprocess.py    # Image loading, camera intrinsics
│       ├── segmentation.py  # SAM-based segmentation
│       ├── scale.py         # Scale detection methods
│       ├── features.py      # SIFT extraction & matching
│       ├── sfm.py           # Structure from Motion
│       ├── dense.py         # Dense reconstruction
│       ├── mesh.py          # Mesh generation
│       ├── volume.py        # Volume calculation
│       └── confidence.py    # Confidence scoring
│
├── tools/                    # User-facing utilities
│   ├── __init__.py
│   ├── client.py            # API test client script
│   └── video_to_images.py   # Video frame extraction script
│
├── scripts/                  # Debug & development scripts
│   ├── __init__.py
│   ├── run_debug_pipeline.py    # Run pipeline with debug output
│   ├── debug_sam.py             # Debug SAM segmentation
│   └── interactive_sam.py       # Interactive SAM GUI tool
│
├── models/                   # SAM model checkpoints
│   └── sam_vit_h.pth
│
├── tests/                    # Unit tests
│   ├── conftest.py          # Pytest fixtures
│   ├── test_api.py
│   ├── test_confidence.py
│   ├── test_features.py
│   ├── test_preprocess.py
│   ├── test_scale.py
│   └── test_volume.py
│
├── data/                     # Sample/test data (gitignored)
│   └── surfboard_frames/    # Sample surfboard images
│
└── docs/                     # Additional documentation
    ├── DEBUG_TOOLS_README.md
    ├── INTERACTIVE_SAM_README.md
    └── debug_tools_plan.md
```

## Configuration

Environment variables:
- `CUDA_VISIBLE_DEVICES` - GPU device(s) to use (default: 0)

Job data retention:
- Jobs and images are retained for 48 hours
- Automatic cleanup runs hourly

## Troubleshooting

**SAM model not found:**
- Ensure the model is downloaded to `models/sam_vit_h.pth` or `models/sam_vit_b.pth`

**Out of memory errors:**
- Use the smaller SAM model (ViT-B)
- Reduce image resolution in `preprocess.py` (default max: 2048px)
- Process fewer images at once

**Low confidence scores:**
- Ensure good image coverage (20-40 images recommended)
- Use consistent lighting
- Include scale reference in images
- Avoid blurry images

**Reconstruction failures:**
- Verify images have sufficient overlap
- Check that the surfboard is clearly visible
- Ensure click point is on the board
