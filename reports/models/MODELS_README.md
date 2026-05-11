# Model Artifacts and Registry

Веса моделей хранятся локально. Current/candidate versions, training data,
validation metrics, drift baselines, and promotion rules are described in
[`../../specs/Monitoring_Retraining.md`](../../specs/Monitoring_Retraining.md).

## Расположение

Все файлы весов должны находиться в:

```text
./src/models/
```

## Файлы моделей

| Filename                    | Назначение                                   |
|-----------------------------|----------------------------------------------|
| `car_detector.pt`           | Детекция автомобиля на изображении           |
| `license_plate_detector.pt` | Детекция номерного знака                     |
| `plate_ocr.pt`              | OCR: распознавание символов номерного знака  |
| `car_embedder.pt`           | Построение эмбеддинга изображения автомобиля |
