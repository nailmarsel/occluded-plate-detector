import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from io import BytesIO
from fastapi.testclient import TestClient
from app.main import app


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self):
        """Test health endpoint returns status."""
        with patch("app.api.routes.health.es_client") as mock_es, \
             patch("app.api.routes.health.s3_client") as mock_s3:
            mock_es.check_health = AsyncMock(return_value=True)
            mock_s3.client = MagicMock()

            response = TestClient(app).get("/api/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ("healthy", "degraded")


class TestSearchEndpoint:
    """Tests for /search endpoint."""

    def test_search_invalid_file_type(self):
        """Test search rejects non-image files."""
        response = TestClient(app).post(
            "/api/v1/search",
            files={"image": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_search_empty_file(self):
        """Test search rejects empty files."""
        response = TestClient(app).post(
            "/api/v1/search",
            files={"image": ("empty.jpg", b"", "image/jpeg")}
        )
        assert response.status_code == 400


class TestIndexEndpoint:
    """Tests for /index endpoint."""

    def test_index_invalid_file_type(self):
        """Test index rejects non-image files."""
        response = TestClient(app).post(
            "/api/v1/index",
            data={"plate_number": "A864AA199"},
            files={"image": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_index_empty_file(self):
        """Test index rejects empty files."""
        response = TestClient(app).post(
            "/api/v1/index",
            data={"plate_number": "A864AA199"},
            files={"image": ("empty.jpg", b"", "image/jpeg")}
        )
        assert response.status_code == 400
