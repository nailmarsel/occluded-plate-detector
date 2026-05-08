# AutobahnCV - Car License Plate Search System

A FastAPI-based application that allows users to search for similar cars by uploading a photo with a partially visible
license plate. The system uses neural networks for car detection, plate recognition, and embedding generation, then
searches Elasticsearch for the top 5 most similar vehicles.

## Architecture

```
User Upload → [YOLO v8 Car Detection] → [YOLO v8 Plate Detection] → [OCR] → [ResNet-108 Embedding] → [MinIO Upload] → [Elasticsearch Search]
```

### Storage Flow

1. **Images** are stored in **MinIO** (S3-compatible object storage)
2. **Metadata** (plate number, embedding, S3 key) is stored in **Elasticsearch**
3. Search results include **presigned URLs** to access images from MinIO

### Neural Network Pipeline

1. **Neuron 1 (YOLO v8)**: Detects and crops the car from the input image
2. **Neuron 2 (YOLO v8)**: Detects and crops the license plate from the car image
3. **Neuron 3 (OCR)**: Recognizes the license plate text from the cropped plate
4. **Neuron 4 (ResNet-108)**: Generates an embedding vector from the cropped car image

### Search Flow

1. Filter Elasticsearch documents by partial plate number match
2. Rank results by embedding similarity (cosine similarity)
3. Return top 5 most similar cars

## Project Structure

```
src/
├── app/                     # Backend (FastAPI)
│   ├── api/
│   │   └── routes/          # API endpoint handlers
│   ├── core/                # Config, logging, enums
│   ├── models/              # Pydantic schemas
│   ├── neurons/             # Neural network wrappers
│   ├── services/            # Pipeline, ES client, S3 client
│   └── main.py
├── frontend/                # Frontend (React + Vite)
│   ├── src/
│   │   ├── components/      # React page components
│   │   ├── services/        # API service layer
│   │   ├── App.jsx          # Router and layout
│   │   └── main.jsx         # Entry point
│   ├── Dockerfile           # Multi-stage build (Node → Nginx)
│   └── nginx.conf           # Nginx config with API proxy
├── tests/                   # Backend tests
├── docker-compose.yml       # Full dev stack
├── requirements.txt
└── README.md
```

The CI/CD workflows for this app live at the repository root in
`.github/workflows/`, because GitHub Actions only runs workflows from that
location.

## Setup & Installation

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (for Elasticsearch + MinIO)
- GPU (recommended for neural network inference)

### 1. Open the app directory

```bash
cd src
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your actual values
```

### 5. Start dependencies with Docker Compose

```bash
# Start Elasticsearch and MinIO (setup jobs run automatically)
docker compose up -d

# Wait ~30 seconds for services to start
# The es-setup and minio-setup jobs run automatically:
docker compose ps
```

**Services available after startup:**
| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | `http://localhost:3000` | React web UI |
| Elasticsearch | `http://localhost:9200` | Vector search + metadata |
| MinIO API | `http://localhost:9000` | S3-compatible image storage |
| MinIO Console | `http://localhost:9001` | Web UI for browsing buckets |
| Kibana | `http://localhost:5601` | ES visualization (optional) |

To start with Kibana:

```bash
docker compose --profile monitoring up -d
```

### 6. Run the full stack (including the app)

```bash
docker compose --profile full up -d
```

This builds and runs the FastAPI app alongside ES and MinIO.

### 7. Prepare Neural Network Models

Update the neuron implementations in `app/neurons/` with your actual models:

- **Neuron 1 & 2**: YOLO v8 models for car and plate detection
- **Neuron 3**: OCR model for plate text recognition
- **Neuron 4**: ResNet-108 for embedding generation

Update model paths in `.env` accordingly. Each neuron class has `TODO` comments showing where to add your code.

## Running the Application

### Development mode (backend only)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Development mode (frontend)

```bash
cd frontend
npm install
npm run dev
```

The frontend dev server runs on `http://localhost:3000` and proxies API requests to the backend.

### Docker (full stack with frontend)

```bash
docker compose --profile full up -d
```

The application will be available at:

- **Frontend (UI)**: `http://localhost:3000`
- **API Base**: `http://localhost:8000/api/v1`
- **Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Documentation

Once running, the API docs are available at:

| Endpoint               | Description                           |
|------------------------|---------------------------------------|
| `/docs`                | Swagger UI — interactive API explorer |
| `/redoc`               | ReDoc — clean static documentation    |
| `/openapi.json`        | Raw OpenAPI 3.1 JSON spec             |
| `/api/v1/openapi/json` | Download OpenAPI JSON spec            |
| `/api/v1/openapi/yaml` | Download OpenAPI YAML spec            |

## API Endpoints

### 1. Search for Similar Cars

**POST** `/api/v1/search`

Upload a car photo with a partially visible license plate to find the top 5 most similar cars.

**Request:** Multipart form with file upload

- `image`: Car photo (JPEG/PNG)

**Response:**

```json
{
  "results": [
    {
      "car_id": "car_123",
      "similarity_score": 0.95,
      "plate_number": "ABC123",
      "image_url": "http://localhost:9000/autobahncv/cars/car_123.jpg?X-Amz-..."
    }
  ],
  "detected_plate": "ABC12",
  "total_found": 3
}
```

### 2. Index a Single Car

**POST** `/api/v1/index`

Add a car to the system. The image is processed, uploaded to MinIO, and indexed in Elasticsearch.

**Request:** Multipart form

- `image`: Car photo (JPEG/PNG)

**Response:**

```json
{
  "car_id": "car_001",
  "plate_number": "ABC123",
  "embedding_dim": 512,
  "status": "indexed"
}
```

### 3. Batch Index from Folder

**POST** `/api/v1/index/batch`

Process all images in a local folder at once.

`prefix` is optional. If you omit it, the backend generates UUID-based `car_id` values.

**Request:** JSON body

```json
{
  "folder_path": "/path/to/car/photos",
  "prefix": "batch_001"
}
```

**Response:**

```json
{
  "total": 10,
  "succeeded": 8,
  "failed": 2,
  "results": [
    {
      "car_id": "batch_001_0",
      "filename": "photo_1.jpg",
      "plate_number": "extracted",
      "status": "indexed"
    },
    {
      "car_id": "batch_001_1",
      "filename": "photo_2.jpg",
      "status": "failed",
      "error": "No car detected in image"
    }
  ]
}
```

### 4. Health Check

**GET** `/api/v1/health`

Check the health status of the application and its dependencies.

**Response:**

```json
{
  "status": "healthy",
  "elasticsearch": true,
  "neurons": {
    "car_detection": "initialized",
    "plate_detection": "initialized",
    "ocr": "initialized",
    "embedding": "initialized"
  }
}
```

## Usage Examples

### Search for similar cars (curl)

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -F "image=@/path/to/car_photo.jpg"
```

### Index a single car (curl)

```bash
curl -X POST "http://localhost:8000/api/v1/index" \
  -F "image=@/path/to/car_photo.jpg"
```

### Batch index from folder (curl)

```bash
curl -X POST "http://localhost:8000/api/v1/index/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "/path/to/car/photos",
    "prefix": "batch_001"
  }'
```

### Using Python requests

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Search
with open("car_photo.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/search",
        files={"image": f}
    )
    print(response.json())

# Index
with open("car_photo.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/index",
        files={"image": f},
        data={"car_id": "car_001"}
    )
    print(response.json())

# Batch index
response = requests.post(
    f"{BASE_URL}/index/batch",
    json={
        "folder_path": "/path/to/car/photos",
        "prefix": "batch_001"
    }
)
print(response.json())
```

## Elasticsearch Index Mapping

The Elasticsearch index must have the following mapping:

```json
{
  "mappings": {
    "properties": {
      "car_id": {
        "type": "keyword"
      },
      "plate_number": {
        "type": "text"
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 512,
        "index": true,
        "similarity": "cosine"
      },
      "s3_key": {
        "type": "keyword"
      },
      "metadata": {
        "type": "object"
      },
      "created_at": {
        "type": "date"
      }
    }
  }
}
```

Run `scripts/setup_elasticsearch.sh` manually if you're not using Docker Compose, or run the `es-setup` job via
`docker compose up es-setup`. Adjust `dims` if your embedding dimension differs from 512.

## CI/CD

### GitHub Actions

| Workflow                               | Trigger                   | What it does                                                                             |
|----------------------------------------|---------------------------|------------------------------------------------------------------------------------------|
| **CI** (`../.github/workflows/ci.yml`) | Push/PR touching `src/**` | Lint (ruff), type check (mypy), run tests, build Docker                                  |
| **CD** (`../.github/workflows/cd.yml`) | Git tag `v*`              | Build & push the AutobahnCV backend Docker image to GHCR, create release, deploy via SSH |

The workflows set `src` as their working directory so app-local commands
such as `pip install -r requirements.txt`, `pytest tests/`, and Docker builds
continue to work after integrating the app into the larger project repository.

### Required GitHub Secrets

```
DEPLOY_HOST          # Server IP/hostname
DEPLOY_USER          # SSH username
DEPLOY_SSH_KEY       # Private SSH key
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_endpoints.py -v
```

## Implementation Notes

### Error Handling

The application uses structured error responses with error codes:

| Code                  | Meaning                            |
|-----------------------|------------------------------------|
| `INVALID_IMAGE`       | Invalid image format or empty file |
| `CAR_NOT_DETECTED`    | No car detected in the image       |
| `PLATE_NOT_DETECTED`  | No license plate detected          |
| `OCR_FAILED`          | OCR processing failed              |
| `EMBEDDING_FAILED`    | Embedding generation failed        |
| `ELASTICSEARCH_ERROR` | Elasticsearch operation failed     |
| `NEURON_ERROR`        | General neural network error       |

### Configuration

All settings are managed via environment variables. See `.env.example` for the full list. Key settings:

- **Elasticsearch**: host, port, credentials, index name
- **MinIO/S3**: endpoint, access/secret keys, bucket name, image prefix
- **Neurons**: model paths, confidence thresholds, embedding dimension
- **Search**: top-k results, similarity threshold

## License

[Your License Here]
