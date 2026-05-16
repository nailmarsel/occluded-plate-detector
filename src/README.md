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

The service monitoring, model monitoring, drift signals, feedback loop, model
registry, and retraining policy are documented in
[`../specs/Monitoring_Retraining.md`](../specs/Monitoring_Retraining.md).

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

This starts dependency services on the Docker network. They do not publish host
ports in the default deployment; the public entry point is the gateway from the
`full` profile.

If you deploy behind a real domain, set `PUBLIC_BASE_URL` before starting the
stack so Grafana and Kibana generate correct redirects:

```bash
PUBLIC_BASE_URL=https://your-domain.example docker compose --profile full up -d
```

The `monitoring` profile starts the dependency and monitoring services without
the app gateway. Use the `full` profile when you want `/app`, `/grafana`, and
`/kibana` to work from the single public port.

### 6. Run the full stack (including the app)

```bash
docker compose --profile full up -d
```

This builds and runs the FastAPI app alongside ES and MinIO.

Only the gateway publishes a host port. Elasticsearch, MinIO, Prometheus,
Grafana, Kibana, MLflow, the FastAPI app, and the frontend stay on the Docker
network.

**Services available through the gateway:**
| Service | URL | Purpose |
|---------|-----|---------|
| App UI | `http://localhost/app/` | React web UI |
| API | `http://localhost/api/v1` | FastAPI API |
| Swagger Docs | `http://localhost/docs` | Interactive API docs |
| Grafana | `http://localhost/grafana/` | Monitoring dashboards |
| Kibana | `http://localhost/kibana/` | ES visualization |
| Metrics | `http://localhost/metrics` | Prometheus metrics endpoint |

### 7. Prepare Neural Network Models

Training workspaces live in `../ml`. Their `make train` targets publish model
artifacts into `src/models`, which Docker mounts into the app container as
`/app/models`:

| Neuron | Training workspace | App artifact |
|--------|--------------------|--------------|
| Car detector | `../ml/car_detector` | `src/models/car_detector.pt` |
| Plate detector | `../ml/plate_detector` | `src/models/license_plate_detector.pt` |
| Plate OCR | `../ml/plate_ocr` | `src/models/plate_ocr.pt` |
| Car embedder | `../ml/car_embedder` | `src/models/car_embedder.pt` |

Configure Docker paths like this:

```env
NEURON1_CAR_DETECTION_MODEL=/app/models/car_detector.pt
NEURON2_PLATE_DETECTION_MODEL=/app/models/license_plate_detector.pt
NEURON3_OCR_MODEL=/app/models/plate_ocr.pt
NEURON4_RESNET_MODEL=/app/models/car_embedder.pt
NEURON4_EMBEDDING_DIM=2048
```

If a training run was completed without `--publish`, copy the workspace
`runs/.../best.pt` checkpoint to the corresponding `src/models/*.pt` artifact
name above.

For Russian license plates, do not rely on the heuristic plate crop fallback in
production. License plates are not part of the COCO classes used by the default
YOLO vehicle model, so `NEURON2_PLATE_DETECTION_MODEL` must point to custom
license-plate detector weights trained or fine-tuned on Russian plate images.
Set `ML_ALLOW_HEURISTIC_PLATE_FALLBACK=False` when you want startup or inference
to fail loudly if those weights are missing.

Recommended detector path:

```env
NEURON2_PLATE_DETECTION_MODEL=/app/models/license_plate_detector.pt
ML_ALLOW_HEURISTIC_PLATE_FALLBACK=False
```

For best Russian results, train or fine-tune an Ultralytics YOLO detector on the
`AY000554/Car_plate_detecting_dataset` Hugging Face dataset. It contains about
25.5K Russian car images with YOLO-format plate boxes and matches this app's
single-class plate detector interface. Generic international plate detectors can
work as a quick smoke test, but they are less reliable on Russian plate layouts,
camera angles, and local image conditions.

Training code lives in `../ml/plate_detector`. Run `make train` there to produce
`models/license_plate_detector.pt` for this Docker app.

To inspect detection quality during search, set:

```env
ML_DEBUG_IMAGE_DIR=/tmp/autobahncv-debug
```

The pipeline will save car and plate crops there. If plate crops do not tightly
contain the number, fix or retrain the detector before tuning OCR.

The OCR post-processing expects Russian private passenger plate structure after
normalization, for example `E507MO136`:

```text
letter + 3 digits + 2 letters + 2-3 digit region
```

If EasyOCR splits the crop into fragments such as `E507MO` and `136`, the
pipeline joins left-to-right fragments and prefers a valid full Russian plate
over a short high-confidence region fragment.

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

- **Frontend (UI)**: `http://localhost/app/`
- **API Base**: `http://localhost/api/v1`
- **Swagger Docs**: `http://localhost/docs`
- **ReDoc**: `http://localhost/redoc`
- **Grafana**: `http://localhost/grafana/`
- **Kibana**: `http://localhost/kibana/`

## API Documentation

Once running, the API docs are available at:

| Endpoint               | Description                           |
|------------------------|---------------------------------------|
| `/docs`                | Swagger UI — interactive API explorer |
| `/redoc`               | ReDoc — clean static documentation    |
| `/openapi.json`        | Raw OpenAPI 3.1 JSON spec             |
| `/api/v1/openapi/json` | Download OpenAPI JSON spec            |
| `/api/v1/openapi/yaml` | Download OpenAPI YAML spec            |
| `/metrics`             | Prometheus metrics endpoint           |

## API Endpoints

### 1. Search for Similar Cars

**POST** `/api/v1/search`

Upload a car photo with a partially visible license plate to find the top 5 most similar cars.

**Request:** Multipart form with file upload

- `image`: Car photo (JPEG/PNG)
- `plate_query` optional: visible plate characters. Use `*` or `?` for each
  hidden character.

Examples:

```text
A8**AA977  -> A8, then two hidden characters, then AA977
A8         -> any indexed plate containing A8
AA977      -> any indexed plate containing AA977
```

For Russian plates, type either Cyrillic or Latin lookalike letters. The app
normalizes `АВЕКМНОРСТУХ` to `ABEKMHOPCTYX` before searching.

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
  "plate_query": "A8**AA977",
  "total_found": 3
}
```

### 2. Index a Single Car

**POST** `/api/v1/index`

Add a car to the system. The image is processed, uploaded to MinIO, and indexed in Elasticsearch.

**Request:** Multipart form

- `image`: Car photo (JPEG/PNG)
- `plate_number`: Known plate number. Required for indexing.

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

### 5. Submit Search Feedback

**POST** `/api/v1/feedback`

Record user feedback for monitoring and later retraining.

**Request:** JSON body

```json
{
  "result_id": "car_123",
  "action": "correct",
  "corrected_plate": "A888AA977",
  "comment": "OCR missed two digits",
  "disputed": false
}
```

Allowed `action` values are `confirm`, `reject`, and `correct`.

## Usage Examples

### Search for similar cars (curl)

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -F "image=@/path/to/car_photo.jpg"
```

With a manually visible plate fragment:

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -F "image=@/path/to/car_photo.jpg" \
  -F "plate_query=A8**AA977"
```

### Index a single car (curl)

```bash
curl -X POST "http://localhost:8000/api/v1/index" \
  -F "image=@/path/to/car_photo.jpg" \
  -F "plate_number=A888AA977"
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
        files={"image": f},
        data={"plate_query": "A8**AA977"}
    )
    print(response.json())

# Index
with open("car_photo.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/index",
        files={"image": f},
        data={"plate_number": "A888AA977"}
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
        "type": "keyword"
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 2048,
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
`docker compose up es-setup`. Adjust `dims` if your embedding dimension differs from 2048.

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
