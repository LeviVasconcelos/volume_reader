"""Debug visualization tools for the reconstruction pipeline."""

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import open3d as o3d


class DebugContext:
    """Manages debug output directory and provides utilities for saving debug artifacts."""

    # Stage name to directory mapping
    STAGE_DIRS = {
        "preprocess": "01_preprocess",
        "segmentation": "02_segmentation",
        "scale": "03_scale",
        "features": "04_features",
        "sfm": "05_sfm",
        "dense": "06_dense",
        "mesh": "07_mesh",
        "volume": "08_volume",
        "confidence": "09_confidence",
        "object_verification": "10_object_verification",
    }

    def __init__(self, output_dir: Path, enabled: bool = True):
        """
        Initialize debug context.

        Args:
            output_dir: Root directory for debug outputs
            enabled: Whether debug output is enabled
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self._stage_dirs: dict[str, Path] = {}

        if enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: str) -> Path:
        """
        Get or create directory for a pipeline stage.

        Args:
            stage: Stage name (e.g., "preprocess", "segmentation")

        Returns:
            Path to the stage directory
        """
        if stage not in self._stage_dirs:
            dir_name = self.STAGE_DIRS.get(stage, stage)
            stage_path = self.output_dir / dir_name
            if self.enabled:
                stage_path.mkdir(parents=True, exist_ok=True)
            self._stage_dirs[stage] = stage_path
        return self._stage_dirs[stage]

    def save_image(
        self,
        stage: str,
        name: str,
        image: np.ndarray,
        quality: int = 95,
    ) -> Optional[Path]:
        """
        Save debug image with consistent naming.

        Args:
            stage: Pipeline stage name
            name: Image filename (without extension)
            image: Image array (BGR or grayscale)
            quality: JPEG quality (1-100)

        Returns:
            Path to saved image, or None if debug disabled
        """
        if not self.enabled:
            return None

        stage_path = self.stage_dir(stage)

        # Determine extension based on image characteristics
        if name.endswith(".png"):
            filepath = stage_path / name
        elif name.endswith(".jpg") or name.endswith(".jpeg"):
            filepath = stage_path / name
        else:
            # Default to PNG for images with alpha or requiring lossless
            # Use JPG for regular color/grayscale images
            filepath = stage_path / f"{name}.jpg"

        # Ensure image is in correct format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        if filepath.suffix.lower() in [".jpg", ".jpeg"]:
            cv2.imwrite(str(filepath), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(str(filepath), image)

        return filepath

    def save_json(
        self,
        stage: str,
        name: str,
        data: dict[str, Any],
    ) -> Optional[Path]:
        """
        Save debug metadata as JSON.

        Args:
            stage: Pipeline stage name
            name: Filename (without .json extension)
            data: Dictionary to save

        Returns:
            Path to saved JSON, or None if debug disabled
        """
        if not self.enabled:
            return None

        stage_path = self.stage_dir(stage)
        if not name.endswith(".json"):
            name = f"{name}.json"
        filepath = stage_path / name

        # Convert numpy arrays and other non-JSON-serializable types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, "__dict__"):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj

        with open(filepath, "w") as f:
            json.dump(convert(data), f, indent=2)

        return filepath

    def save_point_cloud(
        self,
        stage: str,
        name: str,
        cloud: o3d.geometry.PointCloud,
    ) -> Optional[Path]:
        """
        Save point cloud as PLY file.

        Args:
            stage: Pipeline stage name
            name: Filename (without .ply extension)
            cloud: Open3D point cloud

        Returns:
            Path to saved PLY, or None if debug disabled
        """
        if not self.enabled:
            return None

        stage_path = self.stage_dir(stage)
        if not name.endswith(".ply"):
            name = f"{name}.ply"
        filepath = stage_path / name

        o3d.io.write_point_cloud(str(filepath), cloud)

        return filepath

    def save_mesh(
        self,
        stage: str,
        name: str,
        mesh: o3d.geometry.TriangleMesh,
        file_format: str = "ply",
    ) -> Optional[Path]:
        """
        Save mesh as PLY or OBJ file.

        Args:
            stage: Pipeline stage name
            name: Filename (without extension)
            mesh: Open3D triangle mesh
            file_format: Output format ("ply" or "obj")

        Returns:
            Path to saved mesh, or None if debug disabled
        """
        if not self.enabled:
            return None

        stage_path = self.stage_dir(stage)
        ext = f".{file_format}"
        if not name.endswith(ext):
            name = f"{name}{ext}"
        filepath = stage_path / name

        o3d.io.write_triangle_mesh(str(filepath), mesh)

        return filepath

    def save_plot(
        self,
        stage: str,
        name: str,
        fig,
        dpi: int = 150,
    ) -> Optional[Path]:
        """
        Save matplotlib figure as PNG.

        Args:
            stage: Pipeline stage name
            name: Filename (without extension)
            fig: Matplotlib figure
            dpi: Resolution

        Returns:
            Path to saved plot, or None if debug disabled
        """
        if not self.enabled:
            return None

        stage_path = self.stage_dir(stage)
        if not name.endswith(".png"):
            name = f"{name}.png"
        filepath = stage_path / name

        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

        return filepath


# Export all debug functions
from app.pipeline.debug.preprocess_debug import debug_preprocessing
from app.pipeline.debug.segmentation_debug import debug_segmentation
from app.pipeline.debug.scale_debug import debug_scale_detection
from app.pipeline.debug.features_debug import debug_features
from app.pipeline.debug.sfm_debug import debug_sfm
from app.pipeline.debug.dense_debug import debug_dense
from app.pipeline.debug.mesh_debug import debug_mesh
from app.pipeline.debug.volume_debug import debug_volume
from app.pipeline.debug.confidence_debug import debug_confidence
from app.pipeline.debug.object_verification_debug import debug_object_verification

__all__ = [
    "DebugContext",
    "debug_preprocessing",
    "debug_segmentation",
    "debug_scale_detection",
    "debug_features",
    "debug_sfm",
    "debug_dense",
    "debug_mesh",
    "debug_volume",
    "debug_confidence",
    "debug_object_verification",
]
