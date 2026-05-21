# Отчет по валидации MVP моделей - AutobahnCV

## Решение

**MVP validation passed with limitations.**

Этот отчет фиксирует фактические evidence для MVP gate. Он не утверждает, что
все production DoD targets уже достигнуты. Production target для OCR выше
текущего измеренного результата, поэтому финальное решение по production-релизу
остается ограниченным.

## Model artifacts

Полный набор ожидаемых файлов моделей присутствует в `src/models/`:

| Компонент | Artifact |
|-----------|----------|
| Car detector | `src/models/car_detector.pt` |
| License plate detector | `src/models/license_plate_detector.pt` |
| Plate OCR | `src/models/plate_ocr.pt` |
| Car embedder | `src/models/car_embedder.pt` |

SHA256-хэши artifacts сохранены в `reports/models/release_evidence.json`.
Файлы `src/models/*.pt` не хранятся в GitHub repository, потому что они
gitignored как тяжелые бинарные artifacts. Поэтому в CI verifier проверяет
структуру release evidence и governance contracts, а при локальном наличии
файлов дополнительно проверяет размер и SHA256.

Для строгой локальной проверки model binaries используется:

```bash
python3 scripts/verify_release_gates.py --strict-artifacts
```

## Target ML metrics

| Метрика | MVP threshold | Observed | Статус |
|---------|---------------|----------|--------|
| Plate OCR exact match | >= 0.80 | 0.8453 (2405/2845) | Pass |
| Model artifacts present | 4/4 | 4/4 | Pass |
| API/service E2E coverage | present | `src/tests/test_e2e_pipeline.py` | Pass |

Команда для OCR evaluation:

```bash
ml/plate_ocr/.venv/bin/python ml/plate_ocr/scripts/evaluate_plate_ocr.py \
  --batch-size 256 \
  --workers 0 \
  --device cpu
```

Результат:

```text
exact_match=0.8453 (2405/2845)
```

## Production DoD targets

| Метрика | Production target | Текущий статус |
|---------|-------------------|----------------|
| Car detector mAP50 | >= 0.92 | Требуется финальный validation report |
| License plate detector mAP50 | >= 0.90 | Требуется финальный validation report |
| Plate OCR accuracy | >= 0.95 | Ниже target: 0.8453 |
| Car embedder Top-5 accuracy | >= 0.96 | Требуется финальный validation report |

## End-to-end validation

В проект добавлены executable E2E-style API tests для MVP application flow:

- search request с partial plate query и top result;
- single-car indexing с известным номером;
- release gate verification на основе `release_evidence.json`.

Эти тесты используют controlled pipeline/search/storage dependencies, поэтому
проверяют API contract и orchestration без необходимости поднимать
Elasticsearch, MinIO или GPU models во время CI.

## Оставшиеся ограничения

- OCR нужно улучшить перед заявлением production DoD target `0.95`.
- Для production-grade E2E validation все еще нужен фиксированный размеченный
  image-to-top-5 validation dataset.
- Автоматическая замена скрытых символов номера на `*` остается запланированным
  улучшением model/pipeline.
