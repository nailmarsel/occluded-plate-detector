# AutobahnCV API

Этот документ описывает публичный API AutobahnCV для поиска и индексации автомобилей. Разделы health check здесь намеренно не включены.

## Базовый URL

API смонтирован под префиксом:

`/api/v1`

В локальной разработке это обычно означает:

`http://localhost:8000/api/v1`

## Общие соглашения

- Запросы с загрузкой изображений используют формат `multipart/form-data`.
- Поддерживаемые типы изображений: `image/jpeg`, `image/jpg`, `image/png`.
- Ошибки возвращаются в JSON-формате и обычно содержат `error_code`, `message` и иногда `details`.

## Эндпоинты

### Поиск похожих автомобилей

`POST /api/v1/search`

Загрузите фото автомобиля с частично видимым номерным знаком. Сервис обнаружит автомобиль, найдёт номерной знак, выполнит OCR, сгенерирует embedding и выполнит поиск в Elasticsearch по наиболее похожим автомобилям.

#### Запрос

Поле формы:

- `image`: файл изображения автомобиля

#### Ответ

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

#### Примечания

- Эндпоинт возвращает наиболее похожие результаты.
- `image_url` создаётся как presigned URL на основе объекта, сохранённого в S3/MinIO.

### Индексация одного автомобиля

`POST /api/v1/index`

Загрузите фото автомобиля и укажите `car_id`. Изображение будет обработано, загружено в S3-совместимое хранилище и проиндексировано в Elasticsearch.

#### Запрос

Поля формы:

- `image`: файл изображения автомобиля
- `car_id`: уникальный идентификатор автомобиля

#### Ответ

```json
{
  "car_id": "car_001",
  "plate_number": "ABC123",
  "embedding_dim": 512,
  "status": "indexed"
}
```

#### Примечания

- Перед загрузкой сервис обрезает изображение автомобиля.
- В Elasticsearch сохраняются номерной знак, embedding и S3 key.

### Пакетная индексация автомобилей из папки

`POST /api/v1/index/batch`

Обрабатывает все поддерживаемые изображения в локальной папке и индексирует их по одному.

#### Запрос

```json
{
  "folder_path": "/path/to/car/photos",
  "prefix": "batch_001"
}
```

#### Ответ

```json
{
  "total": 10,
  "succeeded": 8,
  "failed": 2,
  "results": [
    {
      "car_id": "batch_001_0",
      "filename": "photo_1.jpg",
      "plate_number": "ABC123",
      "status": "indexed",
      "error": null
    },
    {
      "car_id": "batch_001_1",
      "filename": "photo_2.jpg",
      "plate_number": null,
      "status": "failed",
      "error": "No car detected in image"
    }
  ]
}
```

#### Примечания

- Поддерживаются файлы `.jpg`, `.jpeg`, `.png`.
- Если `prefix` не указан, сервис генерирует `car_id` на основе UUID.

## Ошибки

Типичный формат ошибки:

```json
{
  "detail": {
    "error_code": "INVALID_IMAGE",
    "message": "Invalid image format. Only JPEG and PNG are supported.",
    "details": "optional extra context"
  }
}
```

Типичные коды ошибок API:

- `INVALID_IMAGE`
- `CAR_NOT_DETECTED`
- `PLATE_NOT_DETECTED`
- `OCR_FAILED`
- `EMBEDDING_FAILED`
- `ELASTICSEARCH_ERROR`
- `NEURON_ERROR`

## Примеры запросов

### Поиск

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -F "image=@/path/to/car_photo.jpg"
```

### Индексация

```bash
curl -X POST "http://localhost:8000/api/v1/index" \
  -F "image=@/path/to/car_photo.jpg" \
  -F "car_id=car_001"
```

### Пакетная индексация

```bash
curl -X POST "http://localhost:8000/api/v1/index/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "/path/to/car/photos",
    "prefix": "batch_001"
  }'
```
