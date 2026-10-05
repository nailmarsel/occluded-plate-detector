# Дообучение детектора и распознавателя номеров

Дообучение продолжает обучение **текущих моделей приложения** (`src/models/*.pt`), а не обучает их с нуля,
и строит таблицу «исходная модель vs дообученная» на одних и тех же тестовых данных.

| Модель | Исходный чекпоинт | Что меняется при дообучении | Тестовые наборы для сравнения |
|---|---|---|---|
| Детектор номеров (YOLO11n) | `src/models/license_plate_detector.pt` | lr0 0.001 вместо 0.01, AdamW + cosine; 30 % обучающих кадров получают копию с частично закрытым номером | test как есть + test с закрытой частью каждого номера |
| OCR (CNN + BiLSTM + CTC) | `src/models/plate_ocr.pt` | lr 1e-4 вместо 1e-3, OneCycle-cosine, клиппинг градиента; аугментации: наклон, сдвиг, яркость/контраст, размытие, понижение разрешения | test как есть + test с фиксированным размытием, шумом и низким контрастом |

Почему это должно помочь: исходные модели обучались с лимитом ~25 минут (OCR exact match 0.85 при цели 0.95),
а в датасете детекции почти нет перекрытых номеров, хотя проект именно про них.

Защита от ухудшения:

- OCR сохраняет чекпоинт, **только если он лучше исходной модели на val**.
- Детектор выбирает лучшую эпоху по val, куда входят и чистые, и перекрытые кадры.
- `src/models` не перезаписывается, пока вы явно не попросите (`--publish`). При публикации старая версия
  копируется в `src/models/rollback/`, а модель, ставшая хуже на чистом тесте, не публикуется вовсе.

## 1. Установка (один раз)

Нужен Python 3.11 и несколько ГБ свободного места: датасеты Hugging Face плюс перекрытые копии кадров
(около 0.3 от размера train-выборки детекции).

```bash
cd occluded-plate-detector
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\activate

pip install -r ml/requirements-finetune.txt
```

Для видеокарты NVIDIA сначала поставьте CUDA-сборку PyTorch по инструкции с
[pytorch.org](https://pytorch.org/get-started/locally/), затем ту же команду `pip install`.

Токен Hugging Face необязателен (датасеты публичные), но без него лимиты на скачивание строже:

```bash
export HF_TOKEN=hf_...          # Windows PowerShell: $env:HF_TOKEN="hf_..."
```

## 2. Запуск одной командой

Сначала короткий прогон (~10–15 минут на CPU), чтобы проверить, что всё работает:

```bash
python scripts/finetune_plate_models.py --quick
```

`--quick` = 1 эпоха, до 5 минут на модель, 10 % train детекции, 300 тестовых примеров.
Цифры в таблице после `--quick` — только проверка, не результат.

Полный прогон:

```bash
python scripts/finetune_plate_models.py
```

Скрипт по очереди: скачивает и готовит датасеты → дообучает детектор → дообучает OCR → сравнивает
старые и новые модели → собирает таблицу. На каждую модель отводится до 25 минут обучения
(`--max-minutes`, максимум 30, как и у остальных скриптов обучения в `ml/`).

Полезные варианты:

```bash
python scripts/finetune_plate_models.py --device 0             # NVIDIA GPU
python scripts/finetune_plate_models.py --skip-prepare         # датасеты уже скачаны
python scripts/finetune_plate_models.py --models ocr           # только OCR
python scripts/finetune_plate_models.py --only compare report  # пересравнить уже обученные
python scripts/finetune_plate_models.py --det-batch 8 --ocr-batch 128 --det-epochs 15
python scripts/finetune_plate_models.py --publish              # заменить src/models, если не хуже
python scripts/finetune_plate_models.py --help                 # все параметры
```

На Mac с Apple Silicon обучение идёт на CPU: CTC-loss и обучение YOLO не работают на MPS
(то же ограничение, что и у исходных скриптов обучения).

## 3. Результат

| Файл | Что внутри |
|---|---|
| `reports/models/FINETUNE_COMPARISON.md` | Таблица сравнения: сначала ключевые метрики, затем все, плюс условия обучения |
| `reports/models/finetune/comparison.csv` | Те же данные для Excel (разделитель `;`) |
| `reports/models/finetune/plate_detector.json`, `plate_ocr.json` | Сырые метрики |
| `ml/plate_detector/runs/finetune_plate_yolo/weights/best.pt` | Дообученный детектор |
| `ml/plate_ocr/runs/finetune_plate_ocr/best.pt` | Дообученный OCR (если обогнал исходный на val) |
| `…/finetune_summary.json` | Параметры и длительность дообучения, история эпох OCR |

Метрики в таблице:

- **Детектор:** mAP@0.5 (основная, цель проекта ≥ 0.90), mAP@0.5:0.95, Precision, Recall, F1, время инференса.
- **OCR:** exact match — номер распознан целиком без ошибок (цель ≥ 0.95), точность по символам, CER, время инференса.

Важно для честной интерпретации: аугментации при дообучении похожи на искажения в «сложных» тестовых
наборах, поэтому прирост там ожидаемо больше. Строки с исходным test показывают, что модель не стала хуже
на обычных данных. Test-выборка не участвует ни в исходном обучении, ни в дообучении.

## 4. По шагам через make (macOS / Linux)

```bash
cd ml/plate_detector
make prepare                      # датасет детекции
make finetune DEVICE=cpu          # или DEVICE=0 для GPU
make compare DEVICE=cpu

cd ../plate_ocr
make prepare
make finetune
make compare

cd ../..
python scripts/build_finetune_report.py
```

Параметры make: `FT_EPOCHS`, `FT_LR0`/`FT_LR`, `BATCH`, `IMGSZ`, `FRACTION`, `OCCLUSION`, `MAX_MINUTES`, `LIMIT`.

## 5. Если что-то пошло не так

| Симптом | Что сделать |
|---|---|
| `Prepared dataset not found` / `Missing manifest` | Не выполнена подготовка: запустите без `--skip-prepare` или `make prepare` |
| Ошибка скачивания с Hugging Face (429, 401) | Задайте `HF_TOKEN` |
| Не хватает памяти | Уменьшите `--det-batch` (2) и `--ocr-batch` (32), для детектора — `--det-imgsz 640` |
| OCR: «did not beat the baseline on val» | Дообученная модель не стала лучше — это нормальный исход; попробуйте `--ocr-epochs 12` или GPU |
| Хочу вернуть старую модель после `--publish` | Скопируйте файл из `src/models/rollback/` обратно в `src/models/` |
