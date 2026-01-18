import io
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from app.main import app
from app.models import JobStatus


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    import numpy as np
    import cv2

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


class TestHealthCheck:
    def test_health_check(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "jobs_count" in data


class TestCreateJob:
    def test_create_job_success(self, client, sample_image):
        """Test successful job creation."""
        request_data = {
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

        # Create 3 test images
        files = [
            ("images", ("img1.jpg", io.BytesIO(sample_image), "image/jpeg")),
            ("images", ("img2.jpg", io.BytesIO(sample_image), "image/jpeg")),
            ("images", ("img3.jpg", io.BytesIO(sample_image), "image/jpeg")),
        ]

        with patch('app.main.run_pipeline', new_callable=AsyncMock):
            response = client.post(
                "/jobs",
                files=files,
                data={"request_data": json.dumps(request_data)}
            )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_create_job_too_few_images(self, client, sample_image):
        """Test that at least 3 images are required."""
        request_data = {
            "scale_method": "user_dimensions",
            "user_dimensions": {"length_mm": 1830},
            "board_click_point": {"image_index": 0, "x": 0.5, "y": 0.5}
        }

        files = [
            ("images", ("img1.jpg", io.BytesIO(sample_image), "image/jpeg")),
            ("images", ("img2.jpg", io.BytesIO(sample_image), "image/jpeg")),
        ]

        response = client.post(
            "/jobs",
            files=files,
            data={"request_data": json.dumps(request_data)}
        )

        assert response.status_code == 400
        assert "at least 3 images" in response.json()["detail"].lower()

    def test_create_job_invalid_click_point_index(self, client, sample_image):
        """Test that click point index must be valid."""
        request_data = {
            "scale_method": "user_dimensions",
            "user_dimensions": {"length_mm": 1830},
            "board_click_point": {"image_index": 10, "x": 0.5, "y": 0.5}  # Invalid
        }

        files = [
            ("images", ("img1.jpg", io.BytesIO(sample_image), "image/jpeg")),
            ("images", ("img2.jpg", io.BytesIO(sample_image), "image/jpeg")),
            ("images", ("img3.jpg", io.BytesIO(sample_image), "image/jpeg")),
        ]

        response = client.post(
            "/jobs",
            files=files,
            data={"request_data": json.dumps(request_data)}
        )

        assert response.status_code == 400
        assert "image_index" in response.json()["detail"]

    def test_create_job_invalid_request_data(self, client, sample_image):
        """Test invalid JSON request data."""
        files = [
            ("images", ("img1.jpg", io.BytesIO(sample_image), "image/jpeg")),
            ("images", ("img2.jpg", io.BytesIO(sample_image), "image/jpeg")),
            ("images", ("img3.jpg", io.BytesIO(sample_image), "image/jpeg")),
        ]

        response = client.post(
            "/jobs",
            files=files,
            data={"request_data": "not valid json"}
        )

        assert response.status_code == 400


class TestGetJob:
    def test_get_nonexistent_job(self, client):
        """Test 404 for nonexistent job."""
        response = client.get("/jobs/nonexistent-id")

        assert response.status_code == 404

    def test_get_job_status(self, client, sample_image):
        """Test lightweight status endpoint."""
        # First create a job
        request_data = {
            "scale_method": "user_dimensions",
            "user_dimensions": {"length_mm": 1830},
            "board_click_point": {"image_index": 0, "x": 0.5, "y": 0.5}
        }

        files = [
            ("images", ("img1.jpg", io.BytesIO(sample_image), "image/jpeg")),
            ("images", ("img2.jpg", io.BytesIO(sample_image), "image/jpeg")),
            ("images", ("img3.jpg", io.BytesIO(sample_image), "image/jpeg")),
        ]

        with patch('app.main.run_pipeline', new_callable=AsyncMock):
            create_response = client.post(
                "/jobs",
                files=files,
                data={"request_data": json.dumps(request_data)}
            )

        job_id = create_response.json()["job_id"]

        # Get status
        response = client.get(f"/jobs/{job_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data


class TestDeleteJob:
    def test_delete_nonexistent_job(self, client):
        """Test 404 when deleting nonexistent job."""
        response = client.delete("/jobs/nonexistent-id")

        assert response.status_code == 404

    def test_delete_job_success(self, client, sample_image):
        """Test successful job deletion."""
        # First create a job
        request_data = {
            "scale_method": "user_dimensions",
            "user_dimensions": {"length_mm": 1830},
            "board_click_point": {"image_index": 0, "x": 0.5, "y": 0.5}
        }

        files = [
            ("images", ("img1.jpg", io.BytesIO(sample_image), "image/jpeg")),
            ("images", ("img2.jpg", io.BytesIO(sample_image), "image/jpeg")),
            ("images", ("img3.jpg", io.BytesIO(sample_image), "image/jpeg")),
        ]

        with patch('app.main.run_pipeline', new_callable=AsyncMock):
            create_response = client.post(
                "/jobs",
                files=files,
                data={"request_data": json.dumps(request_data)}
            )

        job_id = create_response.json()["job_id"]

        # Delete job
        response = client.delete(f"/jobs/{job_id}")

        assert response.status_code == 200
        assert "deleted" in response.json()["message"].lower()

        # Verify job is gone
        get_response = client.get(f"/jobs/{job_id}")
        assert get_response.status_code == 404
