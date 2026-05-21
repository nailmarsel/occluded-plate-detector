# Rollback Runbook - AutobahnCV

## Назначение

Документ описывает, как откатить AutobahnCV на предыдущую стабильную версию
моделей или конфигурации, если после релиза падает качество, растет error rate
или система начинает выдавать некорректные результаты.

Runbook закрывает MVP gate по release governance / rollback. Для production его
нужно дополнить конкретными версиями Docker image, ссылками на MLflow runs и
операционными контактами.

## Ответственные

| Роль | Ответственность |
|------|-----------------|
| ML engineer | Проверяет качество модели, сравнивает метрики, подтверждает rollback модели. |
| Backend/DevOps engineer | Меняет model paths / Docker deployment, перезапускает сервис, проверяет health и метрики. |
| Project lead | Принимает финальное решение о rollback или продолжении rollout. |

## Источники evidence

| Artifact | Назначение |
|----------|------------|
| `reports/models/release_evidence.json` | Машинно-проверяемые MVP gates: model artifacts, OCR metric, E2E evidence. |
| `reports/models/model_registry.json` | Политика promotion/rollback и список управляемых моделей. |
| `reports/models/VALIDATION_REPORT.md` | Читаемый отчет по текущей валидации. |
| `scripts/verify_release_gates.py` | Проверка release evidence перед rollout или после rollback. |
| `src/docker-compose.yml` | Текущие model path env vars и mounted `./models:/app/models:ro`. |

## Когда запускать rollback

Rollback запускается вручную, если выполняется хотя бы одно условие:

- `p95 latency > 2s` в течение 5 минут;
- error rate выше 10% в течение 5 минут;
- средний OCR/model confidence падает ниже ожидаемого уровня;
- fallback rate становится выше 50% за 10 минут;
- растет число `reject`, `correct` или `disputed` feedback events;
- новая модель не проходит `scripts/verify_release_gates.py`;
- оператор или project lead фиксирует критический кейс неправильной
  идентификации.

## Перед rollback

1. Зафиксировать время инцидента и текущую версию моделей.
2. Сохранить примеры проблемных запросов: image id, detected plate,
   `plate_query`, top-5 results, confidence, feedback action.
3. Проверить dashboard в Grafana:
   - service latency;
   - error rate;
   - OCR confidence;
   - fallback rate;
   - feedback actions.
4. Запустить локальную проверку release gates:

```bash
python3 scripts/verify_release_gates.py
```

Если verifier падает, rollback обязателен до выяснения причины.

## Как откатить модели

В текущем Docker Compose deployment модели подключаются как read-only volume:

```yaml
volumes:
  - ./models:/app/models:ro
```

Model paths задаются environment variables:

```text
NEURON1_CAR_DETECTION_MODEL=/app/models/car_detector.pt
NEURON2_PLATE_DETECTION_MODEL=/app/models/license_plate_detector.pt
NEURON3_OCR_MODEL=/app/models/plate_ocr.pt
NEURON4_RESNET_MODEL=/app/models/car_embedder.pt
```

### Вариант A: заменить файлы на предыдущие стабильные artifacts

1. Скопировать предыдущие стабильные веса в `src/models/` с теми же именами:

```text
src/models/car_detector.pt
src/models/license_plate_detector.pt
src/models/plate_ocr.pt
src/models/car_embedder.pt
```

2. Обновить `reports/models/release_evidence.json` под стабильные artifacts.
3. Проверить gates:

```bash
python3 scripts/verify_release_gates.py
```

4. Перезапустить app service:

```bash
cd src
docker compose --profile full up -d --build app
```

### Вариант B: переключить model paths на rollback directory

Если стабильные модели хранятся отдельно, например в
`src/models/rollback/`, нужно обновить env vars в `src/docker-compose.yml`:

```text
NEURON1_CAR_DETECTION_MODEL=/app/models/rollback/car_detector.pt
NEURON2_PLATE_DETECTION_MODEL=/app/models/rollback/license_plate_detector.pt
NEURON3_OCR_MODEL=/app/models/rollback/plate_ocr.pt
NEURON4_RESNET_MODEL=/app/models/rollback/car_embedder.pt
```

После этого:

```bash
cd src
docker compose --profile full up -d --build app
```

## Проверка после rollback

1. Проверить, что сервис поднялся:

```bash
curl http://localhost/api/v1/health
```

2. Проверить release gates:

```bash
python3 scripts/verify_release_gates.py
```

3. Запустить backend tests:

```bash
python3 -m pytest src/tests
```

4. Выполнить smoke test поиска через UI или API:

```bash
curl -X POST "http://localhost/api/v1/search" \
  -F "image=@tests/photo_1_2025-10-29_19-54-43.jpg" \
  -F "plate_query=A8**AA977"
```

5. В течение 30 минут следить за Grafana:
   - p95 latency;
   - error rate;
   - OCR confidence;
   - fallback rate;
   - feedback reject/correct/disputed.

## Как вернуть rollout после исправления

1. Подготовить candidate model artifacts.
2. Обновить `reports/models/VALIDATION_REPORT.md`.
3. Обновить `reports/models/release_evidence.json`.
4. Запустить:

```bash
python3 scripts/verify_release_gates.py
python3 -m pytest src/tests
```

5. Получить ручное подтверждение ML engineer и project lead.
6. Перезапустить сервис с candidate artifacts.
7. Мониторить метрики минимум 30 минут.

## Критерий успешного rollback

Rollback считается успешным, если:

- app service работает без startup errors;
- `scripts/verify_release_gates.py` проходит;
- backend tests проходят;
- error rate вернулся ниже 10%;
- p95 latency вернулся ниже 2 секунд;
- feedback reject/correct/disputed не растет аномально;
- оператор подтверждает, что результаты снова пригодны для ручной проверки.
