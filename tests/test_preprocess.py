"""Unit and integration tests for the image preprocessing pipeline and endpoints."""

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.preprocessor import ImagePreprocessor, PreprocessingError

client = TestClient(app)


def create_synthetic_floorplan_with_shadows(width: int = 400, height: int = 300) -> bytes:
    """Create a synthetic floor plan image simulating paper folds and uneven shadows.

    Draws:
    - Uneven lighting gradient (shadow from top-left to bottom-right).
    - Exterior and interior walls (dark lines).
    - Simulated furniture boxes.
    """
    # 1. Base paper with illumination gradient (shadow)
    x = np.linspace(150, 240, width, dtype=np.float32)
    y = np.linspace(140, 255, height, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    paper = ((xv + yv) / 2).astype(np.uint8)
    
    # Convert to 3-channel BGR
    img = cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)

    # 2. Draw outer walls (black rectangle)
    cv2.rectangle(img, (30, 30), (width - 30, height - 30), (20, 20, 20), thickness=6)

    # 3. Draw interior dividing wall
    cv2.line(img, (width // 2, 30), (width // 2, height // 2), (20, 20, 20), thickness=5)

    # 4. Draw simulated furniture rectangle (sofa/bed)
    cv2.rectangle(img, (50, 50), (120, 100), (40, 40, 40), thickness=2)

    # 5. Draw simulated dimension text
    cv2.putText(
        img,
        "4.50m",
        (width // 2 - 30, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (10, 10, 10),
        1,
        cv2.LINE_AA,
    )

    # Encode as PNG bytes
    success, encoded = cv2.imencode(".png", img)
    assert success
    return encoded.tobytes()


# ==========================================
# 1. Unit Tests: ImagePreprocessor Service
# ==========================================

def test_preprocessor_decode_valid_image():
    image_bytes = create_synthetic_floorplan_with_shadows()
    img = ImagePreprocessor.decode_image(image_bytes)
    assert isinstance(img, np.ndarray)
    assert img.shape == (300, 400, 3)


def test_preprocessor_decode_corrupt_data():
    with pytest.raises(PreprocessingError):
        ImagePreprocessor.decode_image(b"this is not an image file")


def test_preprocessor_shadow_removal_and_clahe():
    image_bytes = create_synthetic_floorplan_with_shadows()
    bgr = ImagePreprocessor.decode_image(image_bytes)
    gray = ImagePreprocessor.to_grayscale(bgr)
    
    shadow_free = ImagePreprocessor.remove_shadows(gray)
    assert shadow_free.shape == gray.shape
    # Background should be flattened (higher average brightness in paper areas)
    assert shadow_free.mean() >= 180

    enhanced = ImagePreprocessor.enhance_contrast(shadow_free)
    assert enhanced.shape == gray.shape


def test_preprocessor_binarization():
    image_bytes = create_synthetic_floorplan_with_shadows()
    bgr = ImagePreprocessor.decode_image(image_bytes)
    gray = ImagePreprocessor.to_grayscale(bgr)
    shadow_free = ImagePreprocessor.remove_shadows(gray)

    binary_adapt = ImagePreprocessor.binarize(shadow_free, method="adaptive")
    # Binary images should only contain 0 and 255 values
    unique_vals = set(np.unique(binary_adapt))
    assert unique_vals.issubset({0, 255})


def test_preprocessor_pipeline_detailed():
    image_bytes = create_synthetic_floorplan_with_shadows()
    result = ImagePreprocessor.process_pipeline_detailed(image_bytes)

    assert "metadata" in result
    assert result["metadata"]["width"] == 400
    assert result["metadata"]["height"] == 300
    assert result["metadata"]["total_processing_time_ms"] > 0

    assert "stages_base64" in result
    for stage in ["grayscale", "shadow_free", "contrast_enhanced", "binary"]:
        assert stage in result["stages_base64"]
        assert len(result["stages_base64"][stage]) > 100


# ==========================================
# 2. Integration Tests: FastAPI Endpoints
# ==========================================

def test_endpoint_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "opencv_version" in data
    assert "features" in data


def test_endpoint_preprocess_binary_mode():
    image_bytes = create_synthetic_floorplan_with_shadows()
    files = {"file": ("plano_test.png", io.BytesIO(image_bytes), "image/png")}

    response = client.post("/api/v1/preprocess?filter_mode=binary", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-preprocess-mode"] == "binary"

    # Verify the output is a valid decoded image
    nparr = np.frombuffer(response.content, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    assert decoded is not None
    assert decoded.shape == (300, 400)


def test_endpoint_preprocess_shadow_free_mode():
    image_bytes = create_synthetic_floorplan_with_shadows()
    files = {"file": ("plano_test.jpg", io.BytesIO(image_bytes), "image/jpeg")}

    response = client.post("/api/v1/preprocess?filter_mode=shadow_free", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-preprocess-mode"] == "shadow_free"


def test_endpoint_preprocess_pipeline_details():
    image_bytes = create_synthetic_floorplan_with_shadows()
    files = {"file": ("plano_test.png", io.BytesIO(image_bytes), "image/png")}

    response = client.post("/api/v1/preprocess/pipeline", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "plano_test.png"
    assert "pipeline" in data
    assert data["pipeline"]["metadata"]["width"] == 400
    assert "stages_base64" in data["pipeline"]


def test_endpoint_reject_invalid_extension():
    files = {"file": ("documento.txt", io.BytesIO(b"texto plano"), "text/plain")}
    response = client.post("/api/v1/preprocess", files=files)
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_endpoint_reject_corrupt_image():
    files = {"file": ("corrupto.png", io.BytesIO(b"GIF89a_fake_corrupt"), "image/png")}
    response = client.post("/api/v1/preprocess", files=files)
    assert response.status_code == 400
    assert "Could not decode image" in response.json()["detail"]
