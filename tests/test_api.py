"""
Tests for the Damage.ai FastAPI Assessment API
================================================
Run with: pytest tests/test_api.py -v

These tests verify:
1. Health endpoint returns correct status
2. Assessment endpoint accepts images and returns valid structure
3. Decision trace is always present
4. Invalid inputs are rejected properly
5. Policy endpoint returns configuration
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Import the FastAPI app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api import app


client = TestClient(app)


# ---------------------------------------------------------------------------
# Helper: create a minimal valid JPEG image for testing
# ---------------------------------------------------------------------------

def make_test_image(size_kb: int = 10) -> io.BytesIO:
    """
    Create a minimal valid JPEG image for testing.
    Returns a BytesIO object that can be used as an upload.
    """
    # Minimal JPEG: SOI marker + JFIF header + padding + EOI
    # This creates a technically valid JPEG that the API will accept
    import struct
    
    # Minimal JPEG structure
    soi = b'\xff\xd8'  # Start of image
    # APP0 JFIF marker
    app0 = b'\xff\xe0'
    app0_data = b'\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
    # DQT (quantization table) - minimal
    dqt = b'\xff\xdb\x00\x43\x00'
    dqt_data = bytes(range(1, 65))  # 64 values
    # SOF0 (start of frame) - 1x1 pixel, 1 component
    sof = b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
    # DHT (Huffman table) - minimal DC table
    dht = b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b'
    # SOS (start of scan) + minimal scan data
    sos = b'\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\x7b\x40'
    eoi = b'\xff\xd9'  # End of image
    
    jpeg_bytes = soi + app0 + app0_data + dqt + dqt_data + sof + dht + sos
    # Pad to reach desired size
    padding = b'\x00' * max(0, size_kb * 1024 - len(jpeg_bytes) - 2)
    jpeg_bytes += padding + eoi
    
    return io.BytesIO(jpeg_bytes)


# ---------------------------------------------------------------------------
# Tests: Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    """Tests for GET /health"""
    
    def test_health_returns_200(self):
        """Health endpoint should always return 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self):
        """Health response should have all required fields."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "cv_available" in data
        assert "agent_available" in data
        assert isinstance(data["compare_available"], bool)
        assert "uptime_seconds" in data
    
    def test_health_status_is_healthy(self):
        """System should report healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_version_format(self):
        """Version should be a semantic version string."""
        response = client.get("/health")
        data = response.json()
        parts = data["version"].split(".")
        assert len(parts) == 3, "Version should be in X.Y.Z format"


# ---------------------------------------------------------------------------
# Tests: Assessment endpoint
# ---------------------------------------------------------------------------

class TestAssessment:
    """Tests for POST /assess"""
    
    def test_assess_accepts_jpeg(self):
        """Should accept JPEG images and return 200."""
        image = make_test_image(size_kb=5)
        response = client.post(
            "/assess",
            files={"image": ("test_car.jpg", image, "image/jpeg")},
        )
        assert response.status_code == 200
    
    def test_assess_accepts_png(self):
        """Should accept PNG images and return 200."""
        # Create minimal PNG
        import struct
        import zlib
        
        def make_png():
            # 1x1 red pixel PNG
            signature = b'\x89PNG\r\n\x1a\n'
            
            # IHDR chunk
            ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff)
            ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + ihdr_crc
            
            # IDAT chunk
            raw_data = b'\x00\xff\x00\x00'  # filter byte + RGB
            compressed = zlib.compress(raw_data)
            idat_crc = struct.pack('>I', zlib.crc32(b'IDAT' + compressed) & 0xffffffff)
            idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + idat_crc
            
            # IEND chunk
            iend_crc = struct.pack('>I', zlib.crc32(b'IEND') & 0xffffffff)
            iend = struct.pack('>I', 0) + b'IEND' + iend_crc
            
            # Pad to > 1000 bytes
            padding_data = b'\x00' * 1200
            padding_crc = struct.pack('>I', zlib.crc32(b'tEXt' + padding_data) & 0xffffffff)
            padding_chunk = struct.pack('>I', len(padding_data)) + b'tEXt' + padding_data + padding_crc
            
            return signature + ihdr + idat + padding_chunk + iend
        
        png_bytes = make_png()
        response = client.post(
            "/assess",
            files={"image": ("test_car.png", io.BytesIO(png_bytes), "image/png")},
        )
        assert response.status_code == 200
    
    def test_assess_response_has_assessment_id(self):
        """Every assessment should have a unique ID."""
        image = make_test_image()
        response = client.post(
            "/assess",
            files={"image": ("car.jpg", image, "image/jpeg")},
        )
        data = response.json()
        assert "assessment_id" in data
        assert len(data["assessment_id"]) > 0
    
    def test_assess_response_has_timestamp(self):
        """Response should include ISO timestamp."""
        image = make_test_image()
        response = client.post(
            "/assess",
            files={"image": ("car.jpg", image, "image/jpeg")},
        )
        data = response.json()
        assert "timestamp" in data
        # Should be parseable as ISO format
        from datetime import datetime
        datetime.fromisoformat(data["timestamp"])
    
    def test_assess_response_has_decision(self):
        """Response should include a valid decision."""
        image = make_test_image()
        response = client.post(
            "/assess",
            files={"image": ("car.jpg", image, "image/jpeg")},
        )
        data = response.json()
        assert data["decision"] in ("AUTO_APPROVE", "HUMAN_REVIEW", "ESCALATE")
    
    def test_assess_response_has_decision_trace(self):
        """Every assessment MUST include a decision trace for auditability."""
        image = make_test_image()
        response = client.post(
            "/assess",
            files={"image": ("car.jpg", image, "image/jpeg")},
        )
        data = response.json()
        assert "decision_trace" in data
        assert len(data["decision_trace"]) > 0
        
        # Each trace entry should have required fields
        for trace in data["decision_trace"]:
            assert "rule_applied" in trace
            assert "threshold" in trace
            assert "evidence" in trace
    
    def test_assess_response_has_damages(self):
        """Response should include damages list."""
        image = make_test_image()
        response = client.post(
            "/assess",
            files={"image": ("car.jpg", image, "image/jpeg")},
        )
        data = response.json()
        assert "damages_detected" in data
        assert "total_damages" in data
        assert isinstance(data["damages_detected"], list)
        assert data["total_damages"] == len(data["damages_detected"])
    
    def test_assess_response_has_processing_time(self):
        """Should report processing time in milliseconds."""
        image = make_test_image()
        response = client.post(
            "/assess",
            files={"image": ("car.jpg", image, "image/jpeg")},
        )
        data = response.json()
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], int)
        assert data["processing_time_ms"] >= 0
    
    def test_assess_response_has_model_metadata(self):
        """Response should include model and policy version."""
        image = make_test_image()
        response = client.post(
            "/assess",
            files={"image": ("car.jpg", image, "image/jpeg")},
        )
        data = response.json()
        assert "model_version" in data
        assert "policy_version" in data
        assert "cv_backend" in data
    
    def test_two_assessments_have_different_ids(self):
        """Each assessment should have a unique ID."""
        ids = set()
        for _ in range(3):
            image = make_test_image()
            response = client.post(
                "/assess",
                files={"image": ("car.jpg", image, "image/jpeg")},
            )
            ids.add(response.json()["assessment_id"])
        
        assert len(ids) == 3, "Assessment IDs should be unique"


# ---------------------------------------------------------------------------
# Tests: Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for input validation and error handling."""
    
    def test_reject_non_image_content_type(self):
        """Should reject non-image files."""
        response = client.post(
            "/assess",
            files={"image": ("document.pdf", io.BytesIO(b"x" * 2000), "application/pdf")},
        )
        assert response.status_code == 400
    
    def test_reject_too_small_image(self):
        """Should reject images that are too small (likely corrupt)."""
        tiny_image = io.BytesIO(b'\xff\xd8\xff\xd9')  # minimal JPEG, < 1000 bytes
        response = client.post(
            "/assess",
            files={"image": ("tiny.jpg", tiny_image, "image/jpeg")},
        )
        assert response.status_code == 400
    
    def test_missing_image_returns_422(self):
        """Should return 422 when no image is provided."""
        response = client.post("/assess")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests: paired comparison endpoint
# ---------------------------------------------------------------------------

def make_comparison_image() -> bytes:
    image = Image.new("RGB", (20, 10), color="white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def comparison_uploads(include_previous: bool = True):
    image_bytes = make_comparison_image()
    files = [
        ("initial_image", ("initial.png", image_bytes, "image/png")),
        ("final_image", ("final.png", image_bytes, "image/png")),
    ]
    if include_previous:
        files.append(("previous_image", ("previous.png", image_bytes, "image/png")))
    return files


class TestComparison:
    @pytest.fixture(autouse=True)
    def clear_comparison_token(self, monkeypatch):
        monkeypatch.delenv("DAMAGE_AI_API_TOKEN", raising=False)

    def test_compare_reports_only_unmatched_detections_with_yx_boxes(self, monkeypatch):
        import api

        model_results = [
            {"damages": [
                {"type": "scratch", "severity": "light", "confidence": 0.8, "bbox": [2, 1, 6, 5]},
                {"type": "crack", "severity": "moderate", "confidence": 0.75, "bbox": [4, 4, 14, 9]},
            ]},
            {"damages": [
                {"type": "scratch", "severity": "light", "confidence": 0.82, "bbox": [2, 1, 6, 5]},
                {"type": "crack", "severity": "moderate", "confidence": 0.77, "bbox": [4, 4, 14, 9]},
                {"type": "dent", "severity": "severe", "confidence": 0.91, "bbox": [8, 1, 14, 8]},
            ]},
            {"damages": [
                {"type": "scratch", "severity": "light", "confidence": 0.7, "bbox": [2, 1, 6, 5]},
            ]},
        ]

        class FakeDetector:
            def __init__(self):
                self.results = iter(model_results)

            def detect_damage(self, image):
                return next(self.results)

        monkeypatch.setattr(api, "CV_AVAILABLE", True)
        monkeypatch.setattr(api, "CarDamageDetector", lambda **kwargs: FakeDetector(), raising=False)

        response = client.post(
            "/compare",
            data={"angle": "Frente"},
            files=comparison_uploads(),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["angle"] == "Frente"
        assert [damage["origin"] for damage in data["damages"]] == ["between_trips", "this_trip"]
        assert [damage["damage_type"] for damage in data["damages"]] == ["crack", "dent"]
        assert data["damages"][0]["bbox"] == [0.4, 0.2, 0.9, 0.7]
        assert data["damages"][0]["confidence"] == 0.75
        assert data["damages"][1]["severity"] == "severe"

    def test_comparison_matching_requires_class_and_iou_and_is_one_to_one(self):
        import api

        reference = [{"damage_type": "scratch", "bbox": (0.0, 0.0, 1.0, 1.0)}]
        candidates = [
            {"damage_type": "scratch", "bbox": (0.0, 0.0, 0.5, 1.0)},
            {"damage_type": "scratch", "bbox": (0.0, 0.0, 0.5, 1.0)},
            {"damage_type": "dent", "bbox": (0.0, 0.0, 1.0, 1.0)},
            {"damage_type": "scratch", "bbox": (0.0, 0.0, 0.2, 1.0)},
        ]

        assert api._unmatched_candidate_indexes(reference, candidates) == {1, 2, 3}

    def test_compare_rejects_invalid_angle(self):
        response = client.post(
            "/compare",
            data={"angle": "Top"},
            files=comparison_uploads(include_previous=False),
        )

        assert response.status_code == 400

    def test_compare_accepts_optional_previous_image(self, monkeypatch):
        import api

        class NoDamageDetector:
            def detect_damage(self, image):
                return {"damages": []}

        monkeypatch.setattr(api, "CV_AVAILABLE", True)
        monkeypatch.setattr(api, "CarDamageDetector", lambda **kwargs: NoDamageDetector(), raising=False)
        response = client.post(
            "/compare",
            data={"angle": "Lateral Esquerda"},
            files=comparison_uploads(include_previous=False),
        )

        assert response.status_code == 200
        assert response.json() == {"angle": "Lateral Esquerda", "damages": []}

    def test_compare_requires_configured_token(self, monkeypatch):
        monkeypatch.setenv("DAMAGE_AI_API_TOKEN", "test-secret")
        for headers in ({}, {"Authorization": "Bearer wrong-secret"}):
            response = client.post(
                "/compare",
                data={"angle": "Frente"},
                files=comparison_uploads(include_previous=False),
                headers=headers,
            )

            assert response.status_code == 401
            assert response.headers["www-authenticate"] == "Bearer"
            assert "test-secret" not in response.text

    def test_compare_accepts_configured_token(self, monkeypatch):
        import api

        class NoDamageDetector:
            def detect_damage(self, image):
                return {"damages": []}

        monkeypatch.setenv("DAMAGE_AI_API_TOKEN", "test-secret")
        monkeypatch.setattr(api, "CV_AVAILABLE", True)
        monkeypatch.setattr(api, "CarDamageDetector", lambda **kwargs: NoDamageDetector(), raising=False)
        response = client.post(
            "/compare",
            data={"angle": "Frente"},
            files=comparison_uploads(include_previous=False),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 200
        health_response = client.get("/health")
        assert "test-secret" not in health_response.text


    def test_compare_rejects_invalid_image_content(self):
        response = client.post(
            "/compare",
            data={"angle": "Frente"},
            files=[
                ("initial_image", ("bad.png", b"not an image", "image/png")),
                ("final_image", ("valid.png", make_comparison_image(), "image/png")),
            ],
        )

        assert response.status_code == 400

    def test_compare_fails_explicitly_when_cv_is_unavailable(self, monkeypatch):
        import api

        monkeypatch.setattr(api, "CV_AVAILABLE", False)
        response = client.post(
            "/compare",
            data={"angle": "Traseira"},
            files=comparison_uploads(include_previous=False),
        )

        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]

    def test_compare_returns_500_when_yolo_inference_fails(self, monkeypatch):
        import api

        class FailingDetector:
            def detect_damage(self, image):
                raise RuntimeError("model inference failure")

        monkeypatch.setattr(api, "CV_AVAILABLE", True)
        monkeypatch.setattr(api, "CarDamageDetector", lambda **kwargs: FailingDetector(), raising=False)
        response = client.post(
            "/compare",
            data={"angle": "Lateral Direita"},
            files=comparison_uploads(include_previous=False),
        )

        assert response.status_code == 500
        assert "no comparison results" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Tests: Policy endpoint
# ---------------------------------------------------------------------------

class TestPolicy:
    """Tests for GET /policy"""
    
    def test_policy_returns_200(self):
        """Policy endpoint should return 200."""
        response = client.get("/policy")
        assert response.status_code == 200
    
    def test_policy_has_decision_types(self):
        """Should list all possible decision types."""
        response = client.get("/policy")
        data = response.json()
        assert "decision_types" in data
        assert "AUTO_APPROVE" in data["decision_types"]
        assert "HUMAN_REVIEW" in data["decision_types"]
        assert "ESCALATE" in data["decision_types"]
    
    def test_policy_has_rules_summary(self):
        """Should include a summary of decision rules."""
        response = client.get("/policy")
        data = response.json()
        assert "rules_summary" in data


# ---------------------------------------------------------------------------
# Tests: OpenAPI docs
# ---------------------------------------------------------------------------

class TestDocs:
    """Tests for API documentation availability."""
    
    def test_openapi_json_available(self):
        """OpenAPI spec should be accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Damage.ai — Vehicle Damage Assessment API"
    
    def test_docs_page_available(self):
        """Swagger UI should be accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
