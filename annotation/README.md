# Annotation & Quality Control

Артефакты ручной верификации разметки и контроля качества.

## Структура

```
annotation/
├── label_studio/
│   ├── labeling_config_detection.xml   # конфиг проекта детекции (RectangleLabels)
│   ├── labeling_config_ocr.xml         # конфиг проекта OCR (TextArea)
│   ├── export_detection.json           # пример экспорта JSON-MIN (детекция)
│   └── export_ocr.json                 # пример экспорта JSON-MIN (OCR)
├── scripts/
│   ├── validate_annotations.py         # авто-проверки (acceptance gate)
│   ├── ls_to_yolo.py                   # Label Studio JSON-MIN -> YOLO .txt
│   └── compute_agreement.py            # IoU / exact-match / Cohen's kappa + disputed
├── reports/
│   ├── agreement_summary.json          # сводные метрики согласованности
│   └── disputed_cases.csv              # спорные кейсы + решения арбитра
└── yolo/                               # результат конвертации (генерируется)
```

## Быстрый старт

```bash
pip install   # стандартная библиотека Python, внешних зависимостей нет

python scripts/validate_annotations.py --detection label_studio/export_detection.json
python scripts/validate_annotations.py --ocr       label_studio/export_ocr.json
python scripts/ls_to_yolo.py label_studio/export_detection.json yolo/
python scripts/compute_agreement.py
```

Связанные документы: [`../specs/Annotation_Guidelines.md`](../specs/Annotation_Guidelines.md),
[`../specs/Manual_Labeling_Report.md`](../specs/Manual_Labeling_Report.md).
