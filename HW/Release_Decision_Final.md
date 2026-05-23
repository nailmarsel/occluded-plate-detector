# Статус релиза: approved with limitations

## Текущее состояние

### Готово

**Модели:**
* Все четыре модели обучены и сконфигурированы:
  * `car_detector` (YOLO, confidence threshold 0.25);
  * `license_plate_detector` (YOLO);
  * `plate_ocr` (threshold 0.45, нормализация «1 буква + 3 цифры + 2 буквы + регион», склейка фрагментов);
  * `car_embedder` (ResNet‑50, dims=2048).

**Инфраструктура:**
* CPU inference path поддерживается; GPU может использоваться PyTorch/Ultralytics при наличии подходящего runtime, но Docker Compose не настраивает GPU явно.
* Fallback‑поведение корректно: при `ML_ALLOW_HEURISTIC_PLATE_FALLBACK=False` система честно сообщает об ошибке, а не пытается угадать, где номер.
* Frontend (React 18 + Vite) позволяет:
  * добавлять автомобиль в базу;
  * искать автомобили по фото и номеру.
* Docker Compose поднимает полный стек:
  * Elasticsearch 8.19 с cosine‑similarity и автоматическим созданием индекса (`es-setup`);
  * MinIO с bucket‑настройкой (`minio-setup`);
  * FastAPI‑приложение с непривилегированным пользователем `appuser` и HEALTHCHECK на `/api/v1/health`;


**Наблюдаемость:**
* Реализованы custom Prometheus‑метрики для обработанных изображений, input errors, failures по стадиям, confidence, fallback, similarity score и feedback в `metrics.py`.
* Созданы 3 Grafana‑дашборда (`src/grafana/dashboards/`):
  * AutobahnCV Overview;
  * AutobahnCV ML Pipeline;
  * AutobahnCV Runtime.
* Пороги инцидентов описаны в monitoring docs и rollback runbook; provisioned alert rules в репозитории пока не добавлены.
* Каждый инференс логируется в ES‑индекс `inference-logs`.
* Обратная связь логируется в `feedback-logs`.

**CI и автоматизация:**
* CI запускает `ruff check` и `pytest` на каждый push/PR, который затрагивает `src/**`.
* Валидация входных данных через Pydantic покрыта тестами.
* `scripts/verify_release_gates.py` проверяет release evidence, model registry contract и rollback runbook; MLflow сервис присутствует в Docker Compose.


## Следующие шаги

1. Добавить provisioned alert rules для production-порогов:
   * Error Rate > 10%;
   * p95 > 2 с;
   * Confidence Drop;
   * Fallback Rate > 50%;
   * Low-confidence / empty-plate spikes.
2. Добавить проверку покрытия тестов в CI:
   * Добавить команду `pytest --cov=app --cov-fail-under=85`.
   * Обеспечить покрытие ≥ 85 %.
3. Перед production-деплоем:
   * обновить `reports/models/VALIDATION_REPORT.md` и `release_evidence.json`;
   * получить финальные detector/embedder metrics;
   * проверить rollback artifacts и staging smoke test.
