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
