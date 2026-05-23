# Quality Gates Summary — AutobahnCV

---

## Пройденные gates

| # | Gate | Свидетельство |
|---|------|--------------|
| 1 | **CI/CD pipeline** | `.github/workflows/ci.yml` — два jobs: `lint` и `test` на каждый push/PR в `src/**` |
| 2 | **Линтинг (Ruff)** | `pyproject.toml` настроен (правила E, F, I, N, W, UP); CI запускает `ruff check app/` |
| 3 | **Автоматические тесты (pytest)** | `src/tests/` содержит `test_endpoints.py`, `test_ocr.py`, `test_services.py`; CI запускает `pytest tests/` |
| 4 | **Валидация входных данных** | API отклоняет неверный формат, пустые файлы и некорректный content-type (реализовано и покрыто тестами) |
| 5 | **Контейнеризация** | `Dockerfile` + `docker-compose.yml` с полным стеком (ES, MinIO, Prometheus, Grafana, MLflow, app, frontend, Nginx) |
| 6 | **Мониторинг (Prometheus + Grafana)** | `metrics.py` реализует 10+ метрик; `prometheus.yml` настроен; 3 Grafana-дашборда в `src/grafana/dashboards/` |
| 7 | **Observability / logging** | Логирование в Elasticsearch (индексы `inference-logs`, `feedback-logs`); алерты по error rate, latency, confidence, fallback rate |
| 8 | **Human Feedback Loop** | Эндпоинт `POST /api/v1/feedback`; метрика `autobahncv_feedback_actions_total`; сохранение в ES |
| 9 | **MLflow Model Registry** | MLflow сервис задеплоен в `docker-compose.yml`; политика Staging → Production описана в `specs/Monitoring_Drift_Retraining.md` |
| 10 | **Техническая документация** | `specs/PRD.md`, `specs/DoD.md`, `specs/Data_Spec.md`; `HW/Lekciya-6/` (AI_SYSTEM_DOC, API, INFERENCE_PIPELINE, SLO-SLI, TOOLS) |

---

## Не пройденные gates

| # | Gate | Проблема |
|---|------|---------|
| 1 | **Веса моделей** | Директория `src/models/` **отсутствует**. DoD п.4 требует `car_detector.pt`, `license_plate_detector.pt`, `plate_ocr.pt`, `car_embedder.pt`. Без весов сервис не запускается в production-профиле |
| 2 | **Целевые ML-метрики** | DoD п.3 требует: mAP50 ≥ 0.92 (детектор ТС), mAP50 ≥ 0.90 (номер), accuracy ≥ 0.95 (OCR), Top-5 ≥ 0.96 (эмбеддер). Файлы с результатами валидации в репозитории отсутствуют |
| 3 | **Покрытие тестами ≥ 85%** | DoD п.6 устанавливает порог 85%. В CI нет шага `--cov` и нет enforcement-порога; текущее покрытие не измерено |
| 4 | **Функциональные end-to-end тесты** | DoD п.10 описывает 3 группы тестов (33 образца с реальными изображениями). Тесты не автоматизированы и не входят в CI |

---

## Gates, требующие доработки

| # | Gate | Что не хватает |
|---|------|---------------|
| 1 | **Type checking (mypy)** | `pyproject.toml` содержит конфигурацию mypy, но в CI шага с `mypy app/` **нет**; флаги `disallow_untyped_defs = false` / `no_strict_optional = true` делают проверку нестрогой |
| 2 | **Coverage enforcement в CI** | Тесты запускаются без `pytest-cov`; необходимо добавить `--cov=app --cov-fail-under=85` для соответствия DoD п.6 |
| 3 | **Disaster Recovery** | DoD п.10 требует пошаговую инструкцию по переключению моделей и откату. Документ отсутствует; `ML_STRICT_STARTUP=False` в `docker-compose.yml` — лишь частичная реализация |
