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

---

### Не готово (блокеры)

1. **Директория с весами моделей отсутствует в репозитории:**
   * `src/models/` отсутствует? volume‑монтирование `./models:/app/models:ro` вернёт пустую папку.
   * Файлы `car_detector.pt`, `license_plate_detector.pt`, `plate_ocr.pt`, `car_embedder.pt` необходимо разместить до деплоя.

2. **Несоответствие размерности векторов в Elasticsearch:**
   * Скрипт `setup_elasticsearch.sh` создаёт индекс с `dims=512`, в то время как приложение использует `dims=2048` — запуск скрипта сломает векторный поиск.

3. **Демонстрационные ключи MinIO в конфигурации:**
   * В файлах docker-compose.yml и .env.example используются демонстрационные учётные данные MinIO `Q3AM3UQ867SPQQA43P2F` / `zuf+tfteS...`, опубликованные в официальной документации. Эти ключи общедоступны, что создаёт угрозу безопасности: неавторизованные пользователи могут получить доступ к хранилищу при обнаружении адреса сервера.

4. **Ошибка в передаче даты в Elasticsearch:**
   * В `elasticsearch_client.py` передаётся `None` вместо корректной даты — нужно исправить на `datetime.utcnow().isoformat()`.

---

## Следующий шаг: план устранения блокеров

1. Создать директорию `src/models/` и разместить четыре файла весов. Убедиться, что архитектура моделей совпадает с ожидаемой.
2. Загрузить API‑роутеры FastAPI и проверить:
   * вызов `log_inference_event()` после каждого прохода через пайплайн;
   * отправку Prometheus‑метрик после каждого инференса.
3. Исправить `dims` в `setup_elasticsearch.sh`:
   * Заменить `512` на `2048`.
4. Заменить демонстрационные ключи MinIO:
   * Сгенерировать уникальные значения.
   * Обновить `.env` и `docker-compose.yml`.
5. Добавить проверку покрытия тестов в CI/CD:
   * Добавить команду `pytest --cov=app --cov-fail-under=85`.
   * Обеспечить покрытие ≥ 85 %.
6. Перед деплоем:
   * Запустить `scripts/register_models.py` для всех четырёх моделей с метриками валидации.
   * Перевести модели в стейдж `Production`.
7. Документация:
   * Написать **Disaster Recovery Runbook** (набор сценариев: что случилось → как диагностировать → как исправить → как проверить результат).
   * Создать `specs/Monitoring_Drift_Retraining.md` (документ, отвечающий на вопрос: «Как мы поймём, что модель деградирует, и что тогда делать?»).
