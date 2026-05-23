# Monitoring, Drift & Retraining — AutobahnCV

Документ описывает реализованный контур наблюдаемости и политику drift /
переобучения.
---

## 1. Управляемые модели

| Модель | Workspace | Артефакт | Production-таргет (DoD) |
|--------|-----------|----------|--------------------------|
| `car_detector` | `ml/car_detector` | `src/models/car_detector.pt` | mAP50 ≥ 0.92 |
| `license_plate_detector` | `ml/plate_detector` | `src/models/license_plate_detector.pt` | mAP50 ≥ 0.90 |
| `plate_ocr` | `ml/plate_ocr` | `src/models/plate_ocr.pt` | exact match ≥ 0.95 |
| `car_embedder` | `ml/car_embedder` | `src/models/car_embedder.pt` | Top-5 ≥ 0.96 |

---

## 2. Сервисный мониторинг

- **Метрики (Prometheus):** `http_requests_total`,
  `http_request_duration_seconds`, `http_requests_total{status=~"5.."}`.
- **Расчётные показатели:**
  - **Throughput** = `rate(http_requests_total[1m])`
  - **Error Rate** = `rate(http_requests_total{status=~"5.."}[1m]) / rate(http_requests_total[1m])`
  - **p95 Latency** = `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))`
  - **Availability** = `1 - (rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]))`
- **Где смотреть:** дашборд **AutobahnCV Overview**.

## 3. Мониторинг входных данных

- **Метрики (Prometheus):** `autobahncv_images_processed_total`,
  `autobahncv_input_errors_total`,
  `autobahncv_confidence_car/plate/ocr`, `autobahncv_plate_length`,
  `autobahncv_image_size_bytes`.
- **Логирование:** каждый инференс пишется в индекс Elasticsearch
  `inference-logs` (поля: confidence, plate_length, image_size_bytes, success,
  ошибки).
- **Что отслеживается:** пропуски, пустые значения, некорректные входы,
  изменение распределений (drift).
- **Где смотреть:** дашборды **AutobahnCV ML Pipeline** и **AutobahnCV Runtime**.

## 4. Мониторинг предсказаний модели

- **Метрики:** `autobahncv_plate_fallback_total` (доля эвристического
  определения номера), `autobahncv_search_similarity_score` (гистограмма scores
  поиска), доля low-confidence предсказаний (car confidence < 0.6).
- **Где смотреть:** дашборд **AutobahnCV ML Pipeline** (fallback rate, средний
  similarity score, heatmap распределения).

## 5. Мониторинг бизнес-поведения и Human Feedback Loop

- **API обратной связи:** `POST /api/v1/feedback` — действия `confirm`,
  `reject`, `correct`, плюс флаг `disputed` для спорных кейсов.
- **Метрика:** `autobahncv_feedback_actions_total{action=...}`.
- **Логирование:** индекс `feedback-logs` в Elasticsearch.
- **Где смотреть:** панель **Feedback Actions** на дашборде
  **AutobahnCV ML Pipeline** (количество действий и confirmation rate).
- **Использование:** накопленные `reject` / `correct` / `disputed` кейсы —
  источник новых размеченных примеров для переобучения; экспортируются скриптом
  `scripts/export_feedback.py` (шаг 1 цикла переобучения).

## 6. Сигналы качества модели (proxy-метрики без ground truth)

| Сигнал | Источник | Интерпретация |
|--------|----------|----------------|
| Средний confidence по нейронам | `autobahncv_confidence_*` | падение → деградация детекции/OCR |
| Fallback rate | `autobahncv_plate_fallback_total` | рост → OCR/детектор чаще не справляются |
| Empty plate rate | `inference-logs` | рост → детектор номера пропускает объекты |
| Распределение длины номера | `autobahncv_plate_length` | сдвиг → input drift или OCR drift |
| Similarity score распределение | `autobahncv_search_similarity_score` | сдвиг → embedding/data drift |
| Доля `reject` / `correct` / `disputed` | `feedback-logs` | прямой сигнал качества от операторов |

## 7. Дашборды Grafana

В репозитории provisioned **три** дашборда — файлы лежат в
`src/grafana/dashboards/` и подключаются автоматически через
`src/grafana/provisioning/`:

| Файл | Название | Назначение |
|------|-------|------------|
| `autobahncv-overview.json` | **AutobahnCV Overview** | Application Up, API Request Rate, Images Processed, HTTP Errors, Request Duration, Latency Quantiles |
| `autobahncv-ml-pipeline.json` | **AutobahnCV ML Pipeline** | Neuron Failures (всего и по стадиям), Plate Fallbacks, Avg Search Similarity, Empty Plate Rate, Average Model Confidence, Average Plate Length, Feedback Actions |
| `autobahncv-runtime.json` | **AutobahnCV Runtime** | Prometheus Scrape Up/Duration, App Memory/CPU, File Descriptors, Python GC, Average Uploaded Image Size |

## 8. Алерты

| Алерт | Условие |
|-------|---------|
| High Error Rate | `error_rate > 0.1` (10%) за 5 мин |
| High p95 Latency | `p95 > 2 с` за 5 мин |
| Confidence Drop (drift) | средний confidence car < 0.6 за 10 мин |
| High Fallback Rate | `fallback_rate > 0.5` за 10 мин |
| High Low-Confidence | `low_confidence_rate > 0.3` за 10 мин |

## 9. Инструменты observability

| Задача | Инструмент |
|--------|------------|
| Сбор метрик | `prometheus-client` + `/metrics` endpoint |
| Хранение / агрегация | Prometheus |
| Дашборды | Grafana |
| Drift-профилирование | `scripts/check_drift.py` (PSI по распределениям) + алерты Prometheus |
| Логи инференса / feedback | Elasticsearch (`inference-logs`, `feedback-logs`), просмотр через Kibana |
| Версионирование моделей | MLflow Model Registry |
| Audit / retraining | `scripts/retrain.py` (автоматизированный цикл) |

## 10. Drift-профилирование

Скрипт `scripts/check_drift.py` сравнивает распределения признаков за свежее
окно с зафиксированным baseline и считает **PSI (Population Stability Index)**:

- PSI < 0.10 — дрейфа нет;
- 0.10 ≤ PSI < 0.25 — умеренный дрейф, нужен аудит;
- PSI ≥ 0.25 — сильный дрейф, кандидат на переобучение.

Baseline хранится в `reports/data/drift_baseline.json` и обновляется после
каждого подтверждённого релиза модели. Оперативный drift дополнительно
отслеживается алертами Prometheus (см. §8) и сравнением распределений в индексе
Elasticsearch `inference-logs` за разные временные окна.

## 11. Триггеры аудита

Аудит модели запускается, если выполнено хотя бы одно условие:

- сработал алерт Error Rate / Fallback / Confidence Drop;
- доля low-confidence предсказаний > 30% в течение 1 часа;
- накоплено более 50 `disputed` кейсов в `feedback-logs`;
- `scripts/check_drift.py` показал PSI ≥ 0.10 по любому ключевому признаку;
- плановый ежемесячный аудит.

## 12. Триггеры переобучения

Переобучение запускается, если аудит подтвердил хотя бы одно:

- падение целевой метрики (mAP50 / exact match / Top-5) более чем на 10%
  относительно зафиксированного релиза;
- PSI ≥ 0.25 по входным данным или эмбеддингам;
- накоплено достаточно исправленных операторами примеров (порог по умолчанию —
  500 размеченных образцов), статистически улучшающих метрики на валидации.

Переобучение **не нужно**, если метрики стабильны, алерты не срабатывают,
PSI < 0.10 и новых данных нет, либо кандидат не показывает улучшений.

## 13. Процедура переобучения

Автоматизирована скриптом `scripts/retrain.py`:

1. Экспорт исправленных примеров из `feedback-logs` в обучающий датасет
   (`scripts/export_feedback.py`).
2. Дозапуск подготовки данных (`ml/<workspace>/scripts/prepare_*`).
3. Обучение модели-кандидата (`ml/<workspace>/scripts/train_*`).
4. Оффлайн-валидация (`ml/<workspace>/scripts/evaluate_*`) на фиксированном
   test-split; сравнение метрик кандидата и текущего production.
5. Регистрация кандидата в MLflow в стадии `Staging`
   (`src/scripts/register_models.py`).
6. Обновление `reports/models/release_evidence.json` и
   `reports/models/VALIDATION_REPORT.md`.

## 14. Rollback

Условия и пошаговая процедура отката — в `../HW/ROLLBACK_RUNBOOK.md`.
Триггеры rollback продублированы в `../reports/models/model_registry.json`
(`rollback_triggers`): рост p95 latency / error rate, падение OCR confidence,
рост fallback rate, всплеск `reject` / `disputed`.

## 15. Ответственные

| Роль | Зона ответственности |
|------|----------------------|
| ML-инженер | качество моделей, аудит, подтверждение переобучения |
| Backend/DevOps-инженер | деплой, смена model paths, перезапуск сервиса |
| Project lead | финальное решение о promotion / rollback |
