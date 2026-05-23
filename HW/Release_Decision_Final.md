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
* GPU‑поддержка работает из коробки.
* Fallback‑поведение корректно: при `ML_ALLOW_HEURISTIC_PLATE_FALLBACK=False` система честно сообщает об ошибке, а не пытается угадать, где номер.
* Frontend (React 18 + Vite) позволяет:
  * добавлять автомобиль в базу;
  * искать автомобили по фото и номеру.
* Docker Compose поднимает полный стек:
  * Elasticsearch 8.11 с cosine‑similarity и автоматическим созданием индекса (`es-setup`);
  * MinIO с bucket‑настройкой (`minio-setup`);
  * FastAPI‑приложение с непривилегированным пользователем `appuser` и HEALTHCHECK на `/api/v1/health`;


**Наблюдаемость:**
* Реализованы 10+ Prometheus‑метрик (throughput, error rate, p95 latency, confidence по каждому нейрону, fallback rate, similarity score) в `metrics.py`.
* Созданы 4 Grafana‑дашборда (`src/grafana/dashboards/`):
  * Service Metrics;
  * Input Data;
  * Model Predictions;
  * Quality & Drift.
* Настроены 6 алертов:
  * Error Rate > 10 %;
  * p95 > 2 с;
  * Confidence Drop < 0.6;
  * Fallback Rate > 50 %;
  * Low‑Confidence > 30 %.
* Каждый инференс логируется в ES‑индекс `inference-logs`.
* Обратная связь логируется в `feedback-logs`.

**CI/CD и автоматизация:**
* CI/CD запускает `ruff check` и `pytest` на каждый push.
* Валидация входных данных через Pydantic покрыта тестами.
* Скрипт `scripts/register_models.py` автоматизирует полный цикл регистрации в MLflow: артефакт, метрики, датасет, стейдж Staging → Production.


## Следующие шаги

1. Загрузить API‑роутеры FastAPI и проверить:
   * вызов `log_inference_event()` после каждого прохода через пайплайн;
   * отправку Prometheus‑метрик после каждого инференса.
2. Добавить проверку покрытия тестов в CI/CD:
   * Добавить команду `pytest --cov=app --cov-fail-under=85`.
   * Обеспечить покрытие ≥ 85 %.
3. Перед деплоем:
   * Запустить `scripts/register_models.py` для всех четырёх моделей с метриками валидации.
   * Перевести модели в стейдж `Production`.
4. Документация:
   * Написать **Disaster Recovery Runbook** (набор сценариев: что случилось → как диагностировать → как исправить → как проверить результат).
