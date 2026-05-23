# Inference Pipeline — AutobahnCV

## 1. Выбор режима инференса

Для системы выбран **серверный batch-инференс (Backend Inference)** с доступом
через REST API. Обоснование:

- модели (две YOLOv8, OCR-распознаватель, ResNet-50) требуют GPU/CPU-ресурсов,
  которые нерационально размещать на стороне клиента;
- работа идёт со статичными изображениями, а не с real-time видеопотоком, —
  допустима задержка до P95 ≤ 1.5 c (см. `SLO-SLI.md`);
- централизованный инференс упрощает версионирование моделей, логирование
  каждого предсказания и сбор обратной связи.

Альтернативы (on-device, streaming) отклонены: on-device невозможен из-за
размера моделей и зависимости от индекса Elasticsearch, streaming не нужен,
потому что вход — отдельные кадры.

## 2. Инференс-пайплайн

<img width="1219" height="859" alt="inference pipeline" src="https://github.com/user-attachments/assets/935c803d-aee9-4683-b365-6c880be3e404" />

Пайплайн реализован в `src/app/services/pipeline.py` и оркеструет четыре
ML-«нейрона» из `src/app/neurons/`.

### 2.1 Поток поиска (`POST /api/v1/search`)

| Шаг | Этап | Компонент | Артефакт / выход |
|-----|------|-----------|------------------|
| 1 | Приём и валидация изображения | `api/routes/search.py` | проверка формата (JPEG/PNG), размера, целостности; fail-fast для out-of-domain |
| 2 | Детекция автомобиля | Neuron 1, YOLOv8 (`car_plate_detection.py`) | bounding box авто + confidence |
| 3 | Кроп автомобиля | pipeline | вырезанное изображение авто |
| 4 | Детекция номерного знака | Neuron 2, YOLOv8 | bounding box номера + confidence |
| 5 | OCR номера | Neuron 3, CNN+BiLSTM+CTC (`ocr.py`) | распознанный текст; нечитаемые символы → `*`/`?` |
| 6 | Эмбеддинг автомобиля | Neuron 4, ResNet-50 (`embedding.py`) | вектор 2048-dim |
| 7 | Поиск похожих | `services/elasticsearch_client.py` | kNN по cosine similarity, фильтр по фрагменту номера |
| 8 | Формирование ответа | pipeline | top-5 кандидатов: номер, similarity score, ссылка на изображение |
| 9 | Логирование | `services/inference_logger.py` | запись в индекс `inference-logs`, Prometheus-метрики |

### 2.2 Поток индексации (`POST /api/v1/index`)

Шаги 1–3 и 6 совпадают с потоком поиска. Дополнительно: вырезанное изображение
загружается в MinIO (`services/s3_client.py`), а метаданные и эмбеддинг
индексируются в Elasticsearch с известным номером.

## 3. Обработка ошибок и деградация

- **Невалидный вход** — отклоняется до запуска моделей с понятным `error_code`
  (`INVALID_IMAGE`, см. `API.md`).
- **Автомобиль не найден** — при `ML_ALLOW_FULL_IMAGE_CAR_FALLBACK=True`
  используется полное изображение как кроп; иначе возвращается ошибка.
- **Номер не найден / не распознан** — поиск продолжается только по embedding;
  факт fallback фиксируется метрикой `autobahncv_plate_fallback_total`.
- **Сбой нейрона** — возвращается `NEURON_ERROR`, инцидент логируется.

## 4. Производительность

Целевые показатели задержки по этапам приведены в `SLO-SLI.md`
(detection < 300 мс, OCR < 200 мс, similarity search < 700 мс, E2E P95 ≤ 1.5 c).
Замеры по этапам экспортируются как Prometheus-гистограммы и отображаются на
дашборде **AutobahnCV ML Pipeline**.
