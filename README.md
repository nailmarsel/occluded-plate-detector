# Occluded Plate Detector (AutobahnCV)

A software solution for automatic detection and reconstruction of vehicle
license plates from images. The system works with partially visible or
overlapping plates: it searches for visually similar cars in the indexed
dataset and returns the most likely full license plates.

> Russian version below — [Русская версия].

---

## 1. Project Team

| # | Name | Role |
|---|------|------|
| 1 | Dmitry Baranov | Scrum Master (Documentation & Agile Process) |
| 2 | Nail Siraev | ML Engineering (Model Architecture, Training & MLOps) |
| 3 | Shamil Gafiyatullin | ML Engineering (Model Architecture, Training & MLOps) |
| 4 | Maksim Pishchulin | ML Engineering (Model Architecture, Training & MLOps) |
| 5 | Nepogozhev Daniil | Backend Development & Data Engineering |
| 6 | Maria Isakova | Frontend Development |
| 7 | Daria Sabirova | Backend Development (API Server) |

## 2. Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| Car & plate detection | YOLOv8 (Ultralytics) |
| Plate OCR | Custom CNN + BiLSTM + CTC recognizer (`ml/plate_ocr`); EasyOCR used as runtime fallback |
| Car embedding | ResNet-50 (2048-dim feature vector) |
| Deep learning framework | PyTorch |
| Backend | FastAPI |
| Frontend | React 18 + Vite |
| Vector search & metadata | Elasticsearch 8.x (`dense_vector`, cosine similarity) |
| Object storage | MinIO (S3-compatible) |
| Monitoring | Prometheus + Grafana + Kibana |
| Experiment tracking / registry | MLflow |
| Orchestration | Docker Compose |

## 3. Repository Structure

| Path | Contents |
|------|----------|
| `/specs/` | Product & engineering specs: `PRD.md`, `Data_Spec.md`, `DoD.md`, `Acceptance_Criteria.md`, `Data_Contract.md`, `Annotation_Guidelines.md`, `Monitoring_Drift_Retraining.md` |
| `/src/` | Application code: FastAPI backend, React frontend, Docker Compose, Grafana/Prometheus config |
| `/ml/` | Four model workspaces (`car_detector`, `plate_detector`, `plate_ocr`, `car_embedder`) with train/eval/predict scripts |
| `/notebooks/` | Exploration and training notebooks |
| `/reports/` | `models/` (validation & release evidence) and `data/` (data quality & preprocessing reports) |
| `/scripts/` | Operational scripts: `validate_dataset.py`, `verify_release_gates.py`, `check_drift.py`, `retrain.py` |
| `/HW/` | Course deliverables and release decision documents |
| `/tests/` | Sample test images (full and partially occluded plates) |
| `/.github/workflows/` | CI pipeline (lint, tests, coverage, dataset validation) |

## 4. System Architecture

The system uses a modular pipeline of four ML "neurons" behind a FastAPI layer:

- **Image Input Module** — accepts and validates the uploaded image.
- **Car Detection Module** — YOLOv8 locates the vehicle and crops it.
- **License Plate Detection Module** — YOLOv8 locates the plate region.
- **OCR Module** — recognizes the visible plate text; supports wildcards (`*`, `?`).
- **Embedding Module** — ResNet-50 produces a 2048-dim car embedding.
- **Similarity Search Module** — Elasticsearch kNN search over embeddings.
- **API Layer** — exposes search, indexing, image retrieval and feedback endpoints.

## 5. Observability & Monitoring

The project ships a full observability stack (Prometheus, Grafana,
Elasticsearch). Details, dashboards and the retraining policy are documented in
[`specs/Monitoring_Drift_Retraining.md`](specs/Monitoring_Drift_Retraining.md).

- **Metrics** — `prometheus-fastapi-instrumentator` + `prometheus-client` expose
  service metrics (throughput, latency, error rate) and custom ML metrics
  (per-neuron confidence, fallback rate, plate length, feedback) at `/metrics`.
- **Dashboards** — three provisioned Grafana dashboards in
  `src/grafana/dashboards/`: **AutobahnCV Overview**, **AutobahnCV ML Pipeline**,
  **AutobahnCV Runtime**.
- **Drift** — `scripts/check_drift.py` compares recent inference distributions
  against a stored baseline; operational drift also raises Prometheus alerts.
- **Logging** — every inference and feedback event is stored in the
  Elasticsearch indices `inference-logs` and `feedback-logs`.
- **Retraining** — `scripts/retrain.py` automates the audit → retrain → register
  cycle from accumulated feedback; triggers are defined in `Monitoring_Drift_Retraining.md`.

## 6. Execution Flow

**Indexing flow** (building the reference base):

1. Client sends a car photo to `POST /api/v1/index` with a known plate number.
2. Input validation rejects bad format / size / corrupt files (fail-fast).
3. YOLOv8 detects and crops the car.
4. ResNet-50 produces the 2048-dim embedding.
5. The cropped image is stored in MinIO; metadata + embedding are indexed in Elasticsearch.

**Search flow** (identifying an occluded plate):

1. Client sends a car photo to `POST /api/v1/search` (optional `plate_query`).
2. Input validation runs the same fail-fast checks.
3. YOLOv8 detects the car → crops it → detects the plate region.
4. OCR recognizes the visible plate fragment; unreadable characters stay as `*`/`?`.
5. ResNet-50 produces the query embedding.
6. Elasticsearch performs kNN similarity search, optionally filtered by the plate fragment.
7. The API returns the top-5 candidate cars with similarity scores and stored images.
8. The operator confirms / rejects / corrects the result via `POST /api/v1/feedback`.

A diagram of the inference pipeline is in
[`HW/Lekciya-6/INFERENCE_PIPELINE.md`](HW/Lekciya-6/INFERENCE_PIPELINE.md).

## 7. Quick Start

### 7.1 Prerequisites

- Docker and Docker Compose v2
- Trained model weights placed in `src/models/` (see [`reports/models/MODELS_README.md`](reports/models/MODELS_README.md)):
  `car_detector.pt`, `license_plate_detector.pt`, `plate_ocr.pt`, `car_embedder.pt`

### 7.2 Run the full stack

```bash
git clone https://github.com/nailmarsel/occluded-plate-detector.git
cd occluded-plate-detector/src

# 1. Create the environment file and set your own MinIO credentials
cp .env.example .env
#    edit .env: set MINIO_ROOT_USER / MINIO_ROOT_PASSWORD / S3_ACCESS_KEY / S3_SECRET_KEY

# 2. Put the four .pt weight files into src/models/

# 3. Start the full stack (app, frontend, ES, MinIO, Prometheus, Grafana, Kibana, MLflow)
docker compose --profile full up -d --build
```

### 7.3 Access points

| Service | URL |
|---------|-----|
| Web UI (operator) | http://localhost/ |
| API | http://localhost/api/v1 |
| API health | http://localhost/api/v1/health |
| Grafana | http://localhost/grafana |
| Kibana | http://localhost/kibana |

### 7.4 Local development (backend only)

```bash
cd src
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
---

# Детектор перекрытых номерных знаков (Occluded Plate Detector)

Программное решение для автоматического обнаружения и восстановления
автомобильных номерных знаков по изображениям. Система работает с частично
видимыми или перекрытыми номерами: ищет визуально похожие автомобили в
проиндексированном датасете и возвращает наиболее вероятные полные номера.

## 1. Команда проекта

Состав команды приведён в английской версии выше (раздел 1).

## 2. Технологический стек

| Слой | Технология |
|------|------------|
| Язык | Python 3.11 |
| Детекция авто и номера | YOLOv8 (Ultralytics) |
| OCR номера | Собственный распознаватель CNN + BiLSTM + CTC (`ml/plate_ocr`); EasyOCR — runtime-fallback |
| Эмбеддинг авто | ResNet-50 (вектор размерности 2048) |
| Фреймворк глубокого обучения | PyTorch |
| Backend | FastAPI |
| Frontend | React 18 + Vite |
| Векторный поиск и метаданные | Elasticsearch 8.x (`dense_vector`, косинусное сходство) |
| Объектное хранилище | MinIO (S3-совместимое) |
| Мониторинг | Prometheus + Grafana + Kibana |
| Трекинг экспериментов / реестр | MLflow |
| Оркестрация | Docker Compose |

## 3. Структура репозитория

| Путь | Содержимое |
|------|------------|
| `/specs/` | Спецификации: `PRD.md`, `Data_Spec.md`, `DoD.md`, `Acceptance_Criteria.md`, `Data_Contract.md`, `Annotation_Guidelines.md`, `Monitoring_Drift_Retraining.md` |
| `/src/` | Код приложения: FastAPI-backend, React-frontend, Docker Compose, конфигурация Grafana/Prometheus |
| `/ml/` | Четыре workspace моделей (`car_detector`, `plate_detector`, `plate_ocr`, `car_embedder`) со скриптами train/eval/predict |
| `/notebooks/` | Ноутбуки для исследований и обучения |
| `/reports/` | `models/` (валидация и release evidence) и `data/` (отчёты по качеству и подготовке данных) |
| `/scripts/` | Операционные скрипты: `validate_dataset.py`, `verify_release_gates.py`, `check_drift.py`, `retrain.py` |
| `/HW/` | Учебные материалы и документы по релизному решению |
| `/tests/` | Тестовые изображения (полные и частично перекрытые номера) |
| `/.github/workflows/` | CI-пайплайн (линтинг, тесты, покрытие, валидация датасета) |

## 4. Архитектура системы

Система построена как модульный пайплайн из четырёх ML-«нейронов» за слоем FastAPI:
модуль загрузки изображения, детекция автомобиля (YOLOv8), детекция номера
(YOLOv8), OCR номера (с поддержкой wildcard `*`, `?`), эмбеддинг автомобиля
(ResNet-50), поиск похожих (kNN в Elasticsearch) и API-слой.

## 5. Наблюдаемость и мониторинг

Полный контур наблюдаемости (Prometheus, Grafana, Elasticsearch) описан в
[`specs/Monitoring_Drift_Retraining.md`](specs/Monitoring_Drift_Retraining.md).
Метрики отдаются на `/metrics`; в
`src/grafana/dashboards/` есть три provisioned-дашборда (**AutobahnCV Overview**,
**AutobahnCV ML Pipeline**, **AutobahnCV Runtime**); drift проверяется скриптом
`scripts/check_drift.py`; инференс и feedback логируются в индексы Elasticsearch
`inference-logs` и `feedback-logs`; переобучение автоматизируется скриптом
`scripts/retrain.py`.

## 6. Поток выполнения

Поток индексации и поток поиска подробно описаны в английской версии (раздел 6)
и совпадают с диаграммой в
[`HW/Lekciya-6/INFERENCE_PIPELINE.md`](HW/Lekciya-6/INFERENCE_PIPELINE.md).

## 7. Быстрый старт

```bash
git clone https://github.com/nailmarsel/occluded-plate-detector.git
cd occluded-plate-detector/src

# 1. Создать .env и задать собственные ключи MinIO
cp .env.example .env
#    отредактировать .env: MINIO_ROOT_USER / MINIO_ROOT_PASSWORD / S3_ACCESS_KEY / S3_SECRET_KEY

# 2. Положить четыре файла весов в src/models/

# 3. Поднять полный стек
docker compose --profile full up -d --build
```

Веб-интерфейс оператора — http://localhost/, API — http://localhost/api/v1,
Grafana — http://localhost/grafana, Kibana — http://localhost/kibana.
Подробные шаги, smoke-тест и команды для тестов/линтинга — в английской версии
(разделы 7.1–7.6).
