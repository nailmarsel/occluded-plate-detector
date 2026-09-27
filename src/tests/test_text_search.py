from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.description_parser import has_visual_content, parse_description

TEXT_EMBEDDING = [0.05] * 512


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (64, 64), color=(200, 200, 200))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeTextPipeline:
    def __init__(self, text_available: bool = True):
        self.text_search_available = text_available
        self.encoded_texts: list[str] = []

    async def encode_text(self, text: str) -> list[float]:
        self.encoded_texts.append(text)
        return TEXT_EMBEDDING

    async def encode_car_image(self, image):
        return [0.2] * 512

    async def process_for_search(self, image_data: bytes):
        return {
            "plate_number": "A864AA199",
            "embedding": [0.1] * 2048,
            "cropped_car_image": Image.new("RGB", (32, 32)),
        }

    async def process_for_indexing(self, image_data: bytes, plate_number=None):
        return {
            "plate_number": plate_number,
            "embedding": [0.1] * 2048,
            "clip_embedding": [0.2] * 512,
            "cropped_car_image": Image.new("RGB", (32, 32)),
        }


HITS = {
    "hits": [
        {
            "_source": {
                "car_id": "car_001",
                "plate_number": "A123BC77",
                "s3_key": "cars/car_001.jpg",
            },
            "_score": 0.64,
        }
    ],
    "total": 1,
}


# --- Description parser -----------------------------------------------------


@pytest.mark.parametrize(
    ("description", "plate", "text"),
    [
        ("белая Toyota Camry, номер А1**ВС77", "A1??BC77", "белая Toyota Camry"),
        ("red hatchback A123BC 777", "A123BC777", "red hatchback"),
        (
            "серый седан с номером а 123 вс 77 регион, литые диски",
            "A123BC77",
            "серый седан, литые диски",
        ),
        ("A 123 BC 77 белая лада", "A123BC77", "белая лада"),
        ("красный пикап, г/н К??5ОР, тонированный", "K??5OP", "красный пикап, тонированный"),
        ("white SUV with plate A1_3BC", "A1?3BC", "white SUV"),
        ("номер 777", "777", ""),
        # Model names, years and words are not plates.
        ("чёрный BMW X5 2015 года", "", "чёрный BMW X5 2015 года"),
        ("Mercedes E200 серебристый", "", "Mercedes E200 серебристый"),
        ("номер не виден, синий кроссовер", "", "номер не виден, синий кроссовер"),
        ("", "", ""),
    ],
)
def test_parse_description(description, plate, text):
    parsed = parse_description(description)
    assert parsed.plate_fragment == plate
    assert parsed.text == text


def test_has_visual_content():
    assert has_visual_content("белая камри")
    assert not has_visual_content("")
    assert not has_visual_content("77 , -")


# --- POST /search/text -------------------------------------------------------


def test_text_search_uses_clip_and_plate_filter_from_description():
    pipeline = FakeTextPipeline()
    app.state.processing_pipeline = pipeline

    with patch("app.api.routes.search.es_client") as es_client:
        es_client.search_by_text_embedding = AsyncMock(return_value=HITS)
        response = TestClient(app).post(
            "/api/v1/search/text",
            json={"description": "белая Toyota Camry, номер А1**ВС77"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["search_mode"] == "text"
    assert payload["plate_query"] == "A1??BC77"
    assert payload["text_query"] == "белая Toyota Camry"
    assert payload["results"][0]["plate_number"] == "A123BC77"
    assert payload["results"][0]["image_url"] == "/api/v1/images/cars/car_001.jpg"
    assert pipeline.encoded_texts == ["белая Toyota Camry"]

    kwargs = es_client.search_by_text_embedding.await_args.kwargs
    assert kwargs["text_embedding"] == TEXT_EMBEDDING
    assert kwargs["plate_number"] == "*A1??BC77*"


def test_text_search_explicit_plate_query_overrides_description():
    app.state.processing_pipeline = FakeTextPipeline()

    with patch("app.api.routes.search.es_client") as es_client:
        es_client.search_by_text_embedding = AsyncMock(return_value=HITS)
        response = TestClient(app).post(
            "/api/v1/search/text",
            json={"description": "синий седан, номер А111АА77", "plate_query": "в*7"},
        )

    assert response.status_code == 200
    assert response.json()["plate_query"] == "B*7"
    kwargs = es_client.search_by_text_embedding.await_args.kwargs
    assert kwargs["plate_number"] == "*B?7*"


def test_text_search_with_only_plate_runs_plate_search():
    pipeline = FakeTextPipeline(text_available=False)
    app.state.processing_pipeline = pipeline

    with patch("app.api.routes.search.es_client") as es_client:
        es_client.search_by_plate = AsyncMock(return_value=HITS)
        response = TestClient(app).post(
            "/api/v1/search/text", json={"description": "номер А123ВС77"}
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["search_mode"] == "plate"
    assert payload["text_query"] is None
    assert pipeline.encoded_texts == []
    es_client.search_by_plate.assert_awaited_once()


def test_text_search_rejects_query_without_text_or_plate():
    app.state.processing_pipeline = FakeTextPipeline()
    response = TestClient(app).post(
        "/api/v1/search/text", json={"description": "  , 77 "}
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error_code"] == "INVALID_QUERY"


def test_text_search_returns_503_when_model_unavailable():
    app.state.processing_pipeline = FakeTextPipeline(text_available=False)
    response = TestClient(app).post(
        "/api/v1/search/text", json={"description": "белая камри"}
    )
    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == "TEXT_SEARCH_UNAVAILABLE"


def test_text_search_validates_body():
    response = TestClient(app).post("/api/v1/search/text", json={"description": ""})
    assert response.status_code == 422


# --- POST /search with photo + description ------------------------------------


def test_image_search_with_description_adds_text_clause():
    pipeline = FakeTextPipeline()
    app.state.processing_pipeline = pipeline

    with patch("app.api.routes.search.es_client") as es_client:
        es_client.search_by_plate_and_embedding = AsyncMock(return_value=HITS)
        response = TestClient(app).post(
            "/api/v1/search",
            data={"description": "белый седан, номер А8**АА199"},
            files={"image": ("car.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["search_mode"] == "image+text"
    assert payload["plate_query"] == "A8??AA199"
    assert payload["text_query"] == "белый седан"
    kwargs = es_client.search_by_plate_and_embedding.await_args.kwargs
    assert kwargs["text_embedding"] == TEXT_EMBEDDING
    assert kwargs["plate_number"] == "*A8??AA199*"


def test_image_search_ignores_description_when_model_unavailable():
    app.state.processing_pipeline = FakeTextPipeline(text_available=False)

    with patch("app.api.routes.search.es_client") as es_client:
        es_client.search_by_plate_and_embedding = AsyncMock(return_value=HITS)
        response = TestClient(app).post(
            "/api/v1/search",
            data={"description": "белый седан"},
            files={"image": ("car.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["search_mode"] == "image"
    kwargs = es_client.search_by_plate_and_embedding.await_args.kwargs
    assert "text_embedding" not in kwargs


# --- Indexing and backfill -----------------------------------------------------


def test_index_stores_clip_embedding():
    app.state.processing_pipeline = FakeTextPipeline()

    with (
        patch("app.api.routes.index.es_client") as es_client,
        patch("app.api.routes.index.s3_client") as s3_client,
    ):
        es_client.get_car_by_plate = AsyncMock(return_value=None)
        es_client.index_car = AsyncMock(return_value=True)
        s3_client.upload_image = AsyncMock()
        response = TestClient(app).post(
            "/api/v1/index",
            data={"plate_number": "A123BC77"},
            files={"image": ("car.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    assert es_client.index_car.await_args.kwargs["clip_embedding"] == [0.2] * 512


def test_clip_backfill_updates_cars_without_embedding():
    app.state.processing_pipeline = FakeTextPipeline()
    cars = [
        {"_id": "car_001", "s3_key": "cars/car_001.jpg"},
        {"_id": "car_002", "s3_key": "cars/missing.jpg"},
    ]

    with (
        patch("app.api.routes.index.es_client") as es_client,
        patch("app.api.routes.index.s3_client") as s3_client,
    ):
        es_client.ensure_text_search_mapping = AsyncMock(return_value=True)
        es_client.get_cars_missing_clip_embedding = AsyncMock(
            side_effect=[(cars, 2), ([], 1), ([], 1)]
        )
        es_client.update_clip_embedding = AsyncMock(return_value=True)
        s3_client.download_image = AsyncMock(side_effect=[_jpeg_bytes(), None])
        response = TestClient(app).post("/api/v1/index/clip/backfill?limit=10")

    assert response.status_code == 200
    assert response.json() == {
        "processed": 2,
        "updated": 1,
        "failed": 1,
        "remaining": 1,
    }
    es_client.update_clip_embedding.assert_awaited_once_with("car_001", [0.2] * 512)


def test_clip_backfill_requires_text_model():
    app.state.processing_pipeline = FakeTextPipeline(text_available=False)
    response = TestClient(app).post("/api/v1/index/clip/backfill")
    assert response.status_code == 503


# --- Elasticsearch client --------------------------------------------------------


@pytest.mark.asyncio
async def test_es_hybrid_search_builds_two_boosted_knn_clauses():
    from app.services.elasticsearch_client import ElasticsearchClient

    client = ElasticsearchClient()
    client.client = MagicMock()
    client.client.search = AsyncMock(return_value={"hits": {"hits": [], "total": {"value": 0}}})

    await client.search_by_plate_and_embedding(
        plate_number="*A1??BC77*",
        query_embedding=[0.1] * 2048,
        top_k=5,
        text_embedding=TEXT_EMBEDDING,
        text_weight=0.3,
    )

    knn = client.client.search.await_args.kwargs["body"]["knn"]
    assert isinstance(knn, list) and len(knn) == 2
    image_clause, text_clause = knn
    assert image_clause["field"] == "embedding"
    assert text_clause["field"] == "clip_embedding"
    assert image_clause["boost"] == pytest.approx(0.7)
    assert text_clause["boost"] == pytest.approx(0.3)
    assert text_clause["filter"] == image_clause["filter"]


@pytest.mark.asyncio
async def test_es_image_search_without_text_keeps_single_clause():
    from app.services.elasticsearch_client import ElasticsearchClient

    client = ElasticsearchClient()
    client.client = MagicMock()
    client.client.search = AsyncMock(return_value={"hits": {"hits": [], "total": {"value": 0}}})

    await client.search_by_plate_and_embedding("", [0.1] * 2048, top_k=5)

    knn = client.client.search.await_args.kwargs["body"]["knn"]
    assert isinstance(knn, dict)
    assert "boost" not in knn and "filter" not in knn


@pytest.mark.asyncio
async def test_es_ensure_text_search_mapping_adds_missing_field():
    from app.services.elasticsearch_client import ElasticsearchClient

    client = ElasticsearchClient()
    client.client = MagicMock()
    client.client.indices.exists = AsyncMock(return_value=True)
    client.client.indices.get_mapping = AsyncMock(
        return_value={"cars": {"mappings": {"properties": {"embedding": {}}}}}
    )
    client.client.indices.put_mapping = AsyncMock()

    assert await client.ensure_text_search_mapping() is True
    properties = client.client.indices.put_mapping.await_args.kwargs["properties"]
    assert properties["clip_embedding"]["dims"] == 512
    assert properties["clip_embedding"]["type"] == "dense_vector"


@pytest.mark.asyncio
async def test_es_ensure_text_search_mapping_is_noop_when_present():
    from app.services.elasticsearch_client import ElasticsearchClient

    client = ElasticsearchClient()
    client.client = MagicMock()
    client.client.indices.exists = AsyncMock(return_value=True)
    client.client.indices.get_mapping = AsyncMock(
        return_value={"cars": {"mappings": {"properties": {"clip_embedding": {}}}}}
    )
    client.client.indices.put_mapping = AsyncMock()

    assert await client.ensure_text_search_mapping() is True
    client.client.indices.put_mapping.assert_not_awaited()


# --- CLIP neuron and pipeline wiring (sentence-transformers stubbed) ------------


class _FakeSentenceTransformer:
    def __init__(self, name):
        self.name = name

    def encode(self, items, normalize_embeddings=True, convert_to_numpy=True):
        import numpy as np

        vector = np.ones((len(items), 512), dtype="float32")
        return vector / np.linalg.norm(vector, axis=1, keepdims=True)


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    import sys
    import types

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


@pytest.mark.asyncio
async def test_text_image_neuron_encodes_text_and_image(fake_sentence_transformers):
    from app.neurons.text_image_embedding import TextImageEmbeddingNeuron

    neuron = TextImageEmbeddingNeuron("img-model", "text-model", embedding_dim=512)
    await neuron.initialize()

    text_vector = await neuron.encode_text("белая камри")
    image_vector = await neuron.encode_image(Image.new("RGB", (16, 16)))
    assert len(text_vector) == 512 and len(image_vector) == 512
    assert sum(v * v for v in text_vector) == pytest.approx(1.0, rel=1e-4)
    assert neuron.image_model.name == "img-model"
    assert neuron.text_model.name == "text-model"


@pytest.mark.asyncio
async def test_text_image_neuron_rejects_dim_mismatch(fake_sentence_transformers):
    from app.neurons.text_image_embedding import TextImageEmbeddingNeuron

    neuron = TextImageEmbeddingNeuron("img", "txt", embedding_dim=768)
    with pytest.raises(RuntimeError, match="does not match"):
        await neuron.initialize()


@pytest.mark.asyncio
async def test_pipeline_text_search_degrades_gracefully(monkeypatch):
    import sys

    from app.services.pipeline import ImageProcessingPipeline

    # Without sentence-transformers the text neuron fails to load, but the
    # pipeline keeps working and simply reports text search as unavailable.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    pipeline = ImageProcessingPipeline()
    for name in ("car_detector", "plate_detector", "ocr", "embedding_model"):
        monkeypatch.setattr(pipeline, name, MagicMock(), raising=False)

    from app.neurons.text_image_embedding import TextImageEmbeddingNeuron

    pipeline.text_image_model = TextImageEmbeddingNeuron("img", "txt")
    await pipeline._initialize_neuron("text_image_model", pipeline.text_image_model)

    assert pipeline.text_search_available is False
    assert await pipeline.encode_car_image(Image.new("RGB", (8, 8))) is None
    with pytest.raises(RuntimeError, match="Text search is not available"):
        await pipeline.encode_text("белая камри")
