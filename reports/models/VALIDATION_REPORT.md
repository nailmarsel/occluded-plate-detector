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
Файлы `src/models/*.pt` хранятся непосредственно в Git. Test job в CI проверяет
размер и SHA256 вместе с release evidence и governance contracts.

Для строгой локальной проверки model binaries используется:

```bash
python3 scripts/verify_release_gates.py --strict-artifacts
```

## Target ML metrics

| Метрика | MVP threshold | Observed | Статус |
|---------|---------------|----------|--------|
| Plate OCR exact match | >= 0.80 | 0.8496 (2417/2845) | Pass |
| Model artifacts present | 4/4 | 4/4 | Pass |
| API/service E2E coverage | present | `src/tests/test_e2e_pipeline.py` | Pass |

Команда для OCR evaluation:

```bash
python ml/plate_ocr/scripts/evaluate_plate_ocr.py \
  --workers 2 \
  --device cuda
```

Результат:

```text
exact_match=0.8496 (2417/2845)
```

## Production DoD targets

| Метрика | Production target | Текущий статус |
|---------|-------------------|----------------|
| Car detector mAP50 | >= 0.92 | Требуется финальный validation report |
| License plate detector mAP50 | >= 0.90 | Требуется финальный validation report |
| Plate OCR accuracy | >= 0.95 | Ниже target: 0.8496 |
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
- Backend заменяет неуверенные предсказания на `*` по порогу
  `NEURON3_CONFIDENCE_THRESHOLD`: CRNN использует максимальную softmax-вероятность
  внутри каждого непрерывного CTC-предсказания символа; EasyOCR маскирует
  весь неуверенный текстовый фрагмент, поскольку возвращает confidence фрагмента.
  Если надежно прочитанных символов нет, OCR возвращает ошибку, позволяя поиску
  перейти на embedding-only режим. Маски сохраняются при нормализации номера.
- Это не детектор перекрытий: уверенные ошибочные предсказания и полностью
  пропущенные CTC символы пока не маскируются. CTC blank не обозначает скрытую
  позицию. Нужны размеченные примеры перекрытий, локализация позиций и калибровка
  порога; качество автоматического маскирования на реальных изображениях пока
  не измерено. Изменения относятся к backend, не к training/evaluation CLI.
