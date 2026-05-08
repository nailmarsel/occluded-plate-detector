from enum import Enum


class Status(str, Enum):
    """HTTP status codes as strings."""
    OK = "200 OK"
    CREATED = "201 Created"
    BAD_REQUEST = "400 Bad Request"
    NOT_FOUND = "404 Not Found"
    INTERNAL_ERROR = "500 Internal Server Error"
    SERVICE_UNAVAILABLE = "503 Service Unavailable"


class ErrorCode(str, Enum):
    """Application-specific error codes."""
    INVALID_IMAGE = "INVALID_IMAGE"
    CAR_NOT_DETECTED = "CAR_NOT_DETECTED"
    PLATE_NOT_DETECTED = "PLATE_NOT_DETECTED"
    OCR_FAILED = "OCR_FAILED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    ELASTICSEARCH_ERROR = "ELASTICSEARCH_ERROR"
    NEURON_ERROR = "NEURON_ERROR"
