# AutobahnCV API

Документ описывает текущий публичный API AutobahnCV для поиска, индексации,
получения изображений и сбора обратной связи. Health check endpoints намеренно
не включены.

## Базовый URL

Основной API смонтирован под префиксом:

`/api/v1`

В локальной разработке backend обычно доступен так:

`http://localhost:8000/api/v1`

При запуске полного Docker Compose stack через gateway:

`http://localhost/api/v1`

## Общие соглашения

- Запросы с загрузкой изображений используют `multipart/form-data`.
- Поддерживаемые типы изображений: `image/jpeg`, `image/jpg`, `image/png`.
- Для поиска и индексации поддерживаются российские номера с нормализацией
  кириллических букв в латинские аналоги: `АВЕКМНОРСТУХ` -> `ABEKMHOPCTYX`.
- В запросах частичного номера можно использовать `*` или `?` для скрытых
  символов.
- Ошибки обычно возвращаются в JSON-формате внутри поля `detail`.

Пример ошибки:

```json
{
  "detail": {
    "error_code": "INVALID_IMAGE",
    "message": "Invalid image format. Only JPEG and PNG are supported.",
    "details": "optional extra context"
  }
}
```

## Поиск похожих автомобилей

`POST /api/v1/search`

Загружает фото автомобиля и ищет похожие автомобили в базе. Pipeline выполняет
детекцию автомобиля, детекцию номера, OCR, построение embedding и поиск в
Elasticsearch.

Если пользователь передает `plate_query`, поиск использует этот фрагмент номера.
Если `plate_query` не передан, система пытается использовать номер, найденный
OCR. Если номер не распознан, поиск может продолжиться только по embedding.

### Request

Content-Type: `multipart/form-data`

| Поле | Тип | Обязательное | Описание |
|------|-----|--------------|----------|
| `image` | file | да | Фото автомобиля в JPEG/PNG. |
| `plate_query` | string | нет | Видимый фрагмент номера. Можно использовать `*` или `?`, например `A8**AA977`. |

### Response

```json
{
  "results": [
    {
      "car_id": "car_123",
      "similarity_score": 0.95,
      "plate_number": "A888AA977",
      "image_url": "/api/v1/images/cars/car_123.jpg"
    }
  ],
  "detected_plate": "A8AA977",
  "plate_query": "A8**AA977",
  "total_found": 1
}
```

### Пример

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -F "image=@/path/to/car_photo.jpg" \
  -F "plate_query=A8**AA977"
```

## Индексация одного автомобиля

`POST /api/v1/index`

Добавляет один автомобиль в базу. Backend обрабатывает изображение, генерирует
embedding автомобиля, сохраняет cropped image в MinIO/S3 и индексирует metadata
в Elasticsearch.

В текущей версии `plate_number` обязателен для single indexing. Это снижает
риск записи автомобиля в базу с ошибочным OCR-номером.

### Request

Content-Type: `multipart/form-data`

| Поле | Тип | Обязательное | Описание |
|------|-----|--------------|----------|
| `image` | file | да | Фото автомобиля в JPEG/PNG. |
| `plate_number` | string | да | Известный номер автомобиля, например `A888AA977`. |

### Response

```json
{
  "car_id": "generated-uuid",
  "plate_number": "A888AA977",
  "embedding_dim": 2048,
  "status": "indexed"
}
```

### Пример

```bash
curl -X POST "http://localhost:8000/api/v1/index" \
  -F "image=@/path/to/car_photo.jpg" \
  -F "plate_number=A888AA977"
```

## Пакетная индексация из локальной папки

`POST /api/v1/index/batch`

Индексирует все изображения из локальной папки на стороне backend container /
backend host. Поддерживаются файлы `.jpg`, `.jpeg`, `.png`.

Этот endpoint полезен для локальной подготовки базы, но в production требует
осторожности: путь должен существовать там, где запущен backend.

### Request

Content-Type: `application/json`

| Поле | Тип | Обязательное | Описание |
|------|-----|--------------|----------|
| `folder_path` | string | да | Путь к папке с изображениями. |
| `prefix` | string/null | нет | Префикс для `car_id`. Если не передан, используется UUID. |

```json
{
  "folder_path": "/path/to/car/photos",
  "prefix": "batch_001"
}
```

### Response

```json
{
  "total": 2,
  "succeeded": 1,
  "failed": 1,
  "results": [
    {
      "car_id": "batch_001_0",
      "filename": "A888AA977.jpg",
      "plate_number": "A888AA977",
      "status": "indexed",
      "error": null
    },
    {
      "car_id": "batch_001_1",
      "filename": "bad_image.jpg",
      "plate_number": null,
      "status": "failed",
      "error": "No car detected in image"
    }
  ]
}
```

### Пример

```bash
curl -X POST "http://localhost:8000/api/v1/index/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "/path/to/car/photos",
    "prefix": "batch_001"
  }'
```

## Пакетная индексация из ZIP-архива

`POST /api/v1/index/batch/zip`

Загружает ZIP-архив с изображениями автомобилей и индексирует их. Номер
автомобиля извлекается из имени файла. Например, файл `A864AA199.jpg` будет
проиндексирован с номером `A864AA199`.

Ограничения текущей реализации:

- архив должен быть ZIP;
- максимальный размер архива: 500 MB;
- максимальное количество изображений: 500;
- поддерживаются `.jpg`, `.jpeg`, `.png`;
- служебные файлы `__MACOSX/` игнорируются.

### Request

Content-Type: `multipart/form-data`

| Поле | Тип | Обязательное | Описание |
|------|-----|--------------|----------|
| `archive` | file | да | ZIP-архив с изображениями. |
| `prefix` | string/null | нет | Префикс для `car_id`. Если не передан, используется UUID. |

### Response

Формат ответа такой же, как у `/api/v1/index/batch`.

```json
{
  "total": 2,
  "succeeded": 2,
  "failed": 0,
  "results": [
    {
      "car_id": "zip_0",
      "filename": "A864AA199.jpg",
      "plate_number": "A864AA199",
      "status": "indexed",
      "error": null
    },
    {
      "car_id": "zip_1",
      "filename": "B123CC116.png",
      "plate_number": "B123CC116",
      "status": "indexed",
      "error": null
    }
  ]
}
```

### Пример

```bash
curl -X POST "http://localhost:8000/api/v1/index/batch/zip" \
  -F "archive=@/path/to/cars.zip" \
  -F "prefix=zip"
```

## Получение изображения

`GET /api/v1/images/{object_key}`

Возвращает изображение из MinIO/S3 по object key. Этот endpoint используется
в результатах поиска: поле `image_url` содержит путь вида
`/api/v1/images/cars/<car_id>.jpg`.

### Path parameters

| Параметр | Тип | Описание |
|----------|-----|----------|
| `object_key` | string | S3/MinIO object key. Поддерживает вложенные пути, например `cars/car_123.jpg`. |

### Response

Binary image response:

- `image/jpeg` для JPEG;
- `image/png` для PNG.

Если объект не найден, возвращается `404`:

```json
{
  "detail": "Image not found"
}
```

### Пример

```bash
curl "http://localhost:8000/api/v1/images/cars/car_123.jpg" \
  --output car_123.jpg
```

## Обратная связь по результату

`POST /api/v1/feedback`

Сохраняет действие пользователя по результату модели. Endpoint нужен для
human-in-the-loop процесса, мониторинга качества и будущего retraining.

Допустимые действия:

- `confirm` - пользователь подтвердил результат;
- `reject` - пользователь отклонил результат;
- `correct` - пользователь исправил результат.

### Request

Content-Type: `application/json`

| Поле | Тип | Обязательное | Описание |
|------|-----|--------------|----------|
| `result_id` | string | да | `car_id` или идентификатор результата поиска. |
| `action` | string | да | Одно из значений: `confirm`, `reject`, `correct`. |
| `corrected_plate` | string/null | нет | Исправленный номер, если `action = correct`. |
| `comment` | string/null | нет | Комментарий оператора. |
| `disputed` | boolean | нет | Пометка спорного кейса. По умолчанию `false`. |

```json
{
  "result_id": "car_123",
  "action": "correct",
  "corrected_plate": "A888AA977",
  "comment": "OCR пропустил две цифры",
  "disputed": false
}
```

### Response

```json
{
  "status": "ok"
}
```

### Пример

```bash
curl -X POST "http://localhost:8000/api/v1/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "result_id": "car_123",
    "action": "correct",
    "corrected_plate": "A888AA977",
    "comment": "OCR пропустил две цифры",
    "disputed": false
  }'
```

## Коды ошибок

В API встречаются следующие основные ошибки:

| Код/статус | Где возникает | Причина |
|------------|---------------|---------|
| `INVALID_IMAGE` | `/search`, `/index` | Неподдерживаемый формат изображения или пустой файл. |
| `INVALID_FOLDER` | `/index/batch` | Папка не существует или путь не является папкой. |
| `NO_IMAGES` | `/index/batch`, `/index/batch/zip` | В папке или архиве нет поддерживаемых изображений. |
| `INVALID_ARCHIVE` | `/index/batch/zip` | Файл не является ZIP-архивом. |
| `EMPTY_ARCHIVE` | `/index/batch/zip` | ZIP-архив пустой. |
| `ARCHIVE_TOO_LARGE` | `/index/batch/zip` | Архив больше 500 MB. |
| `TOO_MANY_IMAGES` | `/index/batch/zip` | В архиве больше 500 изображений. |
| `NEURON_ERROR` | `/search`, `/index` | Ошибка ML pipeline: detector, OCR, embedding или indexing flow. |
| `404` | `/images/{object_key}` | Изображение не найдено в хранилище. |
| `400` | `/feedback` | Некорректное значение `action`. |

## Краткая карта API

| Method | Path | Назначение |
|--------|------|------------|
| `POST` | `/api/v1/search` | Найти похожие автомобили по изображению и номеру/фрагменту номера. |
| `POST` | `/api/v1/index` | Проиндексировать один автомобиль с известным номером. |
| `POST` | `/api/v1/index/batch` | Проиндексировать изображения из локальной папки. |
| `POST` | `/api/v1/index/batch/zip` | Проиндексировать изображения из ZIP-архива. |
| `GET` | `/api/v1/images/{object_key}` | Получить сохраненное изображение из MinIO/S3. |
| `POST` | `/api/v1/feedback` | Сохранить обратную связь оператора по результату. |
