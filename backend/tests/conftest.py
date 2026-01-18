import pytest
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_images_dir(tmp_path_factory):
    """Create a temporary directory with test images."""
    import numpy as np
    import cv2

    test_dir = tmp_path_factory.mktemp("test_images")

    # Create several test images
    for i in range(5):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(test_dir / f"image_{i}.jpg"), img)

    return test_dir
