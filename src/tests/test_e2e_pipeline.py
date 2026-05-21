from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (256, 256), color=(180, 180, 180))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class FakePipeline:
    async def process_for_search(self, image_data: bytes):
        assert image_data
        return {
            "plate_number": "A864AA199",
            "embedding": [0.1] * 2048,
            "cropped_car_image": Image.new("RGB", (128, 128)),
            "cropped_plate_image": Image.new("RGB", (64, 24)),
            "car_bbox": [0, 0, 128, 128],
            "plate_bbox": [10, 90, 80, 110],
        }

    async def process_for_indexing(self, image_data: bytes, plate_number=None):
        assert image_data
        return {
            "plate_number": plate_number or "A864AA199",
            "embedding": [0.1] * 2048,
            "cropped_car_image": Image.new("RGB", (128, 128)),
            "cropped_plate_image": Image.new("RGB", (64, 24)),
            "car_bbox": [0, 0, 128, 128],
            "plate_bbox": [10, 90, 80, 110],
        }


def test_search_e2e_returns_top_candidate_for_partial_plate_query():
    app.state.processing_pipeline = FakePipeline()
    search_response = {
        "hits": [
            {
                "_source": {
                    "car_id": "car_001",
                    "plate_number": "A864AA199",
                    "s3_key": "cars/car_001.jpg",
                },
                "_score": 0.97,
            }
        ],
        "total": 1,
    }

    with patch("app.api.routes.search.es_client") as es_client:
        es_client.search_by_plate_and_embedding = AsyncMock(
            return_value=search_response
        )

        response = TestClient(app).post(
            "/api/v1/search",
            data={"plate_query": "A8**AA199"},
            files={"image": ("car.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["detected_plate"] == "A864AA199"
    assert payload["plate_query"] == "A8**AA199"
    assert payload["total_found"] == 1
    assert payload["results"][0]["car_id"] == "car_001"
    assert payload["results"][0]["plate_number"] == "A864AA199"
    assert payload["results"][0]["image_url"] == "/api/v1/images/cars/car_001.jpg"


def test_index_e2e_indexes_known_plate_and_uploads_cropped_car():
    app.state.processing_pipeline = FakePipeline()

    with (
        patch("app.api.routes.index.es_client") as es_client,
        patch("app.api.routes.index.s3_client") as s3_client,
    ):
        es_client.get_car_by_plate = AsyncMock(return_value=None)
        es_client.index_car = AsyncMock(return_value=True)
        s3_client.upload_image = AsyncMock(return_value="cars/car_001.jpg")
        s3_client.delete_image = AsyncMock()

        response = TestClient(app).post(
            "/api/v1/index",
            data={"plate_number": "A864AA199"},
            files={"image": ("car.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plate_number"] == "A864AA199"
    assert payload["embedding_dim"] == 2048
    assert payload["status"] == "indexed"
    assert es_client.index_car.await_count == 1
    assert s3_client.upload_image.await_count == 1
