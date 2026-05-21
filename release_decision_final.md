# Финальное решение по релизу - AutobahnCV

## Решение

**MVP готов. Production-релиз возможен только с ограничениями.**

Проект закрывает основной MVP-сценарий: система принимает изображение
автомобиля, запускает pipeline детекции автомобиля и номерного знака, выполняет
OCR, строит embedding автомобиля, ищет похожие записи в Elasticsearch и
возвращает оператору top candidates.

Система не должна использоваться как полностью автономное решение для
юридически значимой идентификации автомобиля. Результат модели должен
рассматриваться как рекомендация для человека-оператора.

## Что готово

| Блок | Статус |
|------|--------|
| Backend API | Готово для MVP: search, index, batch index, images, feedback, health. |
| Frontend | Готово для MVP: оператор может добавлять автомобили и искать кандидатов. |
| ML pipeline | Готово для MVP: car detector, plate detector, OCR, embedder, Elasticsearch search. |
| Partial plate search | Готово: поддерживаются `plate_query`, `*` и `?`, например `A8**AA977`. |
| Model artifacts | Готово для MVP: ожидаемые artifacts описаны и проверяются через release evidence. |
| Validation evidence | Готово для MVP: добавлен `reports/models/VALIDATION_REPORT.md`. |
| Release gates | Готово для MVP: `scripts/verify_release_gates.py` проходит. |
| E2E-style tests | Готово для MVP: добавлены tests для search/index flow и release gates. |
| Rollback governance | Готово для MVP: добавлен `HW/ROLLBACK_RUNBOOK.md` и `reports/models/model_registry.json`. |
| Observability | Готово для MVP: Prometheus metrics, Grafana dashboards, Elasticsearch logs, feedback logs. |

## Фактические проверки

OCR был проверен на test manifest:

```text
exact_match=0.8453 (2405/2845)
```

Команды проверки:

```bash
python3 scripts/verify_release_gates.py
python3 -m pytest src/tests
```

Для локальной строгой проверки model binaries:

```bash
python3 scripts/verify_release_gates.py --strict-artifacts
```

## Что остается ограничением

1. **OCR ниже production target.**

   Текущий OCR exact match равен `0.8453`, а production DoD target для OCR -
   `0.95`. Для production нужно дообучение и валидация на более сложных
   изображениях.

2. **Автоматическая маскировка скрытых символов еще не завершена.**

   Система поддерживает ручной `plate_query` с `*`, но еще не всегда может сама
   определить закрытую часть номера и заменить ее на `*`.

3. **Нужен production image-to-top-5 validation dataset.**

   E2E-style API tests уже есть, но для production нужен фиксированный
   размеченный набор изображений и автоматическая проверка top-5 результата.

4. **Нужно расширить model registry для production.**

   MVP governance готов, но для production следует добавить реальные MLflow run
   ids, history stable versions и ссылки на artifacts.

## Итог

**Release decision: approved with limitations.**

AutobahnCV готов для демонстрации, MVP-проверки, controlled testing и сбора
feedback. Для production без ограничений нужно улучшить OCR, добавить надежную
автоматическую `*`-маскировку закрытых символов, подготовить production E2E
validation dataset и расширить model governance.
