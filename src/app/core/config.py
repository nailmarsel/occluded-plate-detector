from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "AutobahnCV - Car License Plate Search"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"
    ML_STRICT_STARTUP: bool = False
    ML_ALLOW_HEURISTIC_PLATE_FALLBACK: bool = False
    ML_DEBUG_IMAGE_DIR: str = ""

    # Elasticsearch
    ELASTICSEARCH_HOST: str = "localhost"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_INDEX: str = "cars"
    ELASTICSEARCH_USERNAME: str = ""
    ELASTICSEARCH_PASSWORD: str = ""

    # S3 / Garage Storage
    S3_ENDPOINT_URL: str = "http://localhost:8076"
    S3_ACCESS_KEY: str = "Q3AM3UQ867SPQQA43P2F"
    S3_SECRET_KEY: str = "zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG"
    S3_BUCKET_NAME: str = "autobahncv"
    S3_REGION: str = "us-east-1"
    S3_IMAGE_PREFIX: str = "cars/"
    S3_USE_SSL: bool = False

    # Neuron 1: YOLO - Car detection and cropping
    NEURON1_CAR_DETECTION_MODEL: str = "yolo26n.pt"
    NEURON1_CONFIDENCE_THRESHOLD: float = 0.5

    # Neuron 2: YOLO - License plate detection and cropping
    NEURON2_PLATE_DETECTION_MODEL: str = "license_plate_detector.pt"
    NEURON2_CONFIDENCE_THRESHOLD: float = 0.5

    # Neuron 3: OCR - License plate text recognition
    NEURON3_OCR_MODEL: str = "en,ru"
    NEURON3_CONFIDENCE_THRESHOLD: float = 0.45

    # Neuron 4: ResNet - Embedding generation
    NEURON4_RESNET_MODEL: str = "resnet50"
    NEURON4_EMBEDDING_DIM: int = 2048

    # Search
    SEARCH_TOP_K: int = 5
    EMBEDDING_SIMILARITY_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
