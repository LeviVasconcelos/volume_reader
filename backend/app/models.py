from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ScaleMethod(str, Enum):
    ARUCO = "aruco"
    CREDIT_CARD = "credit_card"
    USER_DIMENSIONS = "user_dimensions"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class ClickPoint(BaseModel):
    image_index: int = Field(..., description="Index of the image where user clicked")
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized x coordinate (0-1)")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized y coordinate (0-1)")


class BoundingBox(BaseModel):
    """Bounding box for SAM box prompt (normalized coordinates 0-1)."""
    x_min: float = Field(..., ge=0.0, le=1.0, description="Left edge")
    y_min: float = Field(..., ge=0.0, le=1.0, description="Top edge")
    x_max: float = Field(..., ge=0.0, le=1.0, description="Right edge")
    y_max: float = Field(..., ge=0.0, le=1.0, description="Bottom edge")

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert to pixel coordinates (x1, y1, x2, y2)."""
        return (
            int(self.x_min * width),
            int(self.y_min * height),
            int(self.x_max * width),
            int(self.y_max * height),
        )


class BoardLandmarks(BaseModel):
    """
    Key landmark points on the surfboard for multi-point SAM prompts.
    All coordinates are normalized (0-1). Set to None if not visible.
    """
    nose: Optional[tuple[float, float]] = Field(None, description="Tip of the nose (x, y)")
    tail: Optional[tuple[float, float]] = Field(None, description="Center of the tail (x, y)")
    rail_left: Optional[tuple[float, float]] = Field(None, description="Point on left rail (x, y)")
    rail_right: Optional[tuple[float, float]] = Field(None, description="Point on right rail (x, y)")
    center: Optional[tuple[float, float]] = Field(None, description="Center of the board (x, y)")

    def get_points(self) -> list[tuple[float, float]]:
        """Return list of all defined points."""
        points = []
        for pt in [self.nose, self.tail, self.rail_left, self.rail_right, self.center]:
            if pt is not None:
                points.append(pt)
        return points


class SegmentationPrompt(BaseModel):
    """
    Flexible segmentation prompt supporting multiple input types.
    SAM will use all provided prompts together for best results.
    """
    image_index: int = Field(default=0, description="Reference image index")

    # Point prompts (foreground points on the board)
    click_point: Optional[ClickPoint] = Field(None, description="Single click point (legacy)")
    landmarks: Optional[BoardLandmarks] = Field(None, description="Board landmark points")
    extra_points: list[tuple[float, float]] = Field(default_factory=list, description="Additional foreground points")

    # Negative points (background - NOT the board)
    background_points: list[tuple[float, float]] = Field(default_factory=list, description="Background points (not board)")

    # Box prompt
    bounding_box: Optional[BoundingBox] = Field(None, description="Bounding box around the board")

    def get_all_foreground_points(self) -> list[tuple[float, float]]:
        """Get all foreground point prompts."""
        points = []

        if self.click_point:
            points.append((self.click_point.x, self.click_point.y))

        if self.landmarks:
            points.extend(self.landmarks.get_points())

        points.extend(self.extra_points)

        return points


class ArucoScaleData(BaseModel):
    aruco_id: int = Field(..., description="ArUco marker ID to detect")
    aruco_size_mm: float = Field(..., gt=0, description="Physical size of marker in mm")


class CreditCardScaleData(BaseModel):
    card_type: str = Field(default="standard", description="Card type (standard = 85.6x53.98mm)")


class UserDimensionsScaleData(BaseModel):
    length_mm: Optional[float] = Field(None, gt=0, description="Board length in mm")
    width_mm: Optional[float] = Field(None, gt=0, description="Board width in mm")
    thickness_mm: Optional[float] = Field(None, gt=0, description="Board thickness in mm")


class JobCreateRequest(BaseModel):
    scale_method: ScaleMethod
    aruco_data: Optional[ArucoScaleData] = None
    credit_card_data: Optional[CreditCardScaleData] = None
    user_dimensions: Optional[UserDimensionsScaleData] = None
    board_click_point: ClickPoint


class Dimensions(BaseModel):
    length_mm: float
    width_mm: float
    thickness_mm: float


class JobResult(BaseModel):
    volume_liters: float
    dimensions: Dimensions
    confidence: float = Field(..., ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: Optional[str] = None
    result: Optional[JobResult] = None
    errors: list[str] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: Optional[str] = None
