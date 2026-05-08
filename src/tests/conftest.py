import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import os
import sys

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_pipeline():
    """Mock ImageProcessingPipeline with fake neuron responses."""
    with patch("app.api.routes.index.ImageProcessingPipeline") as MockPipeline:
        instance = MockPipeline.return_value
        instance.process_for_indexing = AsyncMock(
            return_value={
                "plate_number": "ABC123",
                "embedding": [0.1] * 512,
                "cropped_car_image": MagicMock(),
                "cropped_plate_image": MagicMock(),
                "car_bbox": [0, 0, 100, 100],
                "plate_bbox": [20, 20, 80, 40],
            }
        )
        yield instance


@pytest.fixture
def mock_s3_client():
    """Mock S3 client."""
    with patch("app.api.routes.index.s3_client") as mock:
        mock.upload_image = AsyncMock(return_value="cars/test_car.jpg")
        yield mock


@pytest.fixture
def mock_es_client():
    """Mock Elasticsearch client."""
    with patch("app.api.routes.index.es_client") as mock:
        mock.index_car = AsyncMock(return_value=True)
        mock.search_by_plate_and_embedding = AsyncMock(
            return_value={
                "hits": [
                    {
                        "_source": {
                            "car_id": "car_001",
                            "plate_number": "ABC123",
                            "s3_key": "cars/car_001.jpg",
                        },
                        "_score": 0.95,
                    }
                ],
                "total": 1,
            }
        )
        yield mock


@pytest.fixture
def sample_image_bytes():
    """Generate a minimal valid JPEG."""
    # 1x1 red pixel JPEG
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01"
        b"\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07"
        b"\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13"
        b"\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' \",#\x1c\x1c"
        b"(7teletext7(teletext\xa0\xff\xc0\x00\x0b\x08\x00\x01\x00"
        b"\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01"
        b"\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01"
        b"\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10"
        b"\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00"
        b"\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"
        b"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t"
        b"\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghi"
        b"jstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95"
        b"\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa"
        b"\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6"
        b"\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1"
        b"\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5"
        b"\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?"
        b"\x00\xfb\xd4\xff\xd9"
    )
