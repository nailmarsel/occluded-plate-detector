from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "AutobahnCV - Car License Plate Search"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"

    # Elasticsearch
    ELASTICSEARCH_HOST: str = "localhost"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_INDEX: str = "cars"
    ELASTICSEARCH_USERNAME: str = "elastic"
    ELASTICSEARCH_PASSWORD: str = "changeme"

    # S3 / Garage Storage
    S3_ENDPOINT_URL: str = "http://localhost:8076"
    S3_ACCESS_KEY: str = "Q3AM3UQ867SPQQA43P2F"
    S3_SECRET_KEY: str = "zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG"
    S3_BUCKET_NAME: str = "autobahncv"
    S3_REGION: str = "us-east-1"
    S3_IMAGE_PREFIX: str = "cars/"
    S3_USE_SSL: bool = False

    # Neuron 1: YOLO v8 - Car detection and cropping
    NEURON1_CAR_DETECTION_MODEL: str = "yolov8x.pt"
    NEURON1_CONFIDENCE_THRESHOLD: float = 0.5

    # Neuron 2: YOLO v8 - License plate detection and cropping
    NEURON2_PLATE_DETECTION_MODEL: str = "yolov8_plate.pt"
    NEURON2_CONFIDENCE_THRESHOLD: float = 0.5

    # Neuron 3: OCR - License plate text recognition
    NEURON3_OCR_MODEL: str = "ocr_model"
    NEURON3_CONFIDENCE_THRESHOLD: float = 0.6

    # Neuron 4: ResNet-108 - Embedding generation
    NEURON4_RESNET_MODEL: str = "resnet108_embedding.pt"
    NEURON4_EMBEDDING_DIM: int = 512

    # Search
    SEARCH_TOP_K: int = 5
    EMBEDDING_SIMILARITY_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
