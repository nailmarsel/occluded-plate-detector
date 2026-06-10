# Manual Labeling & Quality Control Report

Отчёт по раунду ручной верификации разметки
---

## 1. Объём работ

| Параметр | Детекция | OCR |
| :--- | :--- | :--- |
| Инструмент | Label Studio (RectangleLabels) | Label Studio (TextArea) |
| Конфиг | `labeling_config_detection.xml` | `labeling_config_ocr.xml` |
| Проверено вручную | 2 564 (весь test-сплит) | 2 845 (весь test-сплит) |
| Double-annotation (10%) | 256 | 455 |
| Аннотаторы | Nail, Maksim | Nail, Maksim |
| Reviewer | Shamil | Shamil |
| Arbiter | Nail Siraev (Lead) | Nail Siraev (Lead) |

---

## 2. Метрики согласованности (Inter-Annotator Agreement)

Рассчитаны скриптом `scripts/compute_agreement.py`, сводка —
`reports/agreement_summary.json`.

### 2.1 Детекция

| Метрика | Значение | Порог приёмки |
| :--- | :--- | :--- |
| Средний IoU между аннотаторами | **0.881** | ≥ 0.85 |
| Медианный IoU | 0.902 | — |
| Доля пар с IoU ≥ 0.85 | **79.7%** | — |
| Спорных кейсов (IoU < 0.85) | 52 | — |

### 2.2 OCR

| Метрика | Значение | Порог приёмки |
| :--- | :--- | :--- |
| Exact Match Agreement (строка целиком) | **97.6%** | ≥ 95% |
| Char-level agreement | 99.73% | — |
| Cohen's κ (символьное согласие) | **0.997** | ≥ 0.80 |
| Спорных кейсов (несовпадение строки) | 11 | — |

**Вывод:** согласованность по OCR — отличная (κ ≈ 1.0). По детекции средний
IoU выше порога, но ~20% боксов требуют уточнения — это типично для «плотных»
рамок номера; такие кейсы прошли арбитраж.

---

## 3. Контроль качества (Quality Control)

* **Авто-проверки** (`scripts/validate_annotations.py`): диапазон координат,
  положительная площадь bbox, соответствие OCR-строки регексу
  `^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$`, алфавит символов. Все
  принятые задачи проходят проверки без ошибок (acceptance gate).
* **Двойная слепая разметка** на 10% выборки для расчёта IAA (см. §2).
* **Review reviewer'ом:** задачи с расхождениями или флагами
  (`ambiguous`, `occlusion=heavy`, `glare`, `dirty_plate`) возвращались на
  доработку с комментарием.

---

## 4. Спорные кейсы и арбитраж (Disputed Cases)

Все расхождения выгружены в `reports/disputed_cases.csv` (63 кейса:
52 детекция + 11 OCR). По каждому зафиксировано финальное решение арбитра в
колонке `final_decision`.

Типовые причины расхождений:

| Причина | Задача | Решение |
| :--- | :--- | :--- |
| «Свободная» vs «плотная» рамка | детекция | принят плотный bbox по правилу §2.2 гайдлайна |
| Перекрытие >70% (нечитаемый знак) | детекция | помечен `occlusion=heavy`, исключён из train |
| Путаница похожих символов (O↔0, B↔8, T↔1) | OCR | арбитр сверял с увеличенным кропом, фиксировал верный символ |

После арбитража исправленные метки вошли в **gold-набор**, который
используется как эталон для оценки моделей детекции и OCR.

---

## 5. Воспроизводимость

```bash
# 1. авто-проверки разметки
python scripts/validate_annotations.py --detection label_studio/export_detection.json
python scripts/validate_annotations.py --ocr       label_studio/export_ocr.json

# 2. конвертация Label Studio -> YOLO
python scripts/ls_to_yolo.py label_studio/export_detection.json yolo/

# 3. расчёт согласованности + выгрузка спорных кейсов
python scripts/compute_agreement.py
```

Все скрипты детерминированы (`seed=119`).
