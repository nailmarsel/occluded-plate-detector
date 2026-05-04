# SLI/SLO 

## Целевые показатели производительности и надежности

Для системы детекции и восстановления перекрытых номерных знаков установлены следующие Service Level Indicators (SLI) и Service Level Objectives (SLO):

---

## 1. Latency SLO

End-to-End Processing Time (от загрузки изображения до возврата топ-5 результатов):
- P95: < 1500 мс
- P99: < 2000 мс
Latency по ключевым компонентам (P95):
- Vehicle & Plate Detection (YOLOv8): < 300 мс
- OCR Processing (Tesseract): < 200 мс
- Similarity Search (ResNet-108 + ElasticSearch): < 700 мс

---

## 2. Availability SLO (Доступность системы)

API Uptime: ≥ 99.5% 

---

## 3. Model Accuracy SLO (Точность моделей)

### Детекция и OCR:
- Vehicle Detector (YOLOv8): mAP50 ≥ 0.92
- License Plate Localization (YOLOv8): mAP50 ≥ 0.90
- OCR (Tesseract): Full Plate Accuracy ≥ 0.95

### Поиск похожих автомобилей:
- Top-5 Accuracy (ResNet-108): ≥ 0.96
- Top-1 Accuracy для полных номеров: ≥ 0.85
- Top-5 Accuracy для перекрытых номеров: ≥ 0.80

### Функциональные тесты:
- Полные номера (10 образцов): 100% найдены в Top-1 с confidence > 0.95
- Частично скрытые (15 образцов): ≥ 93% найдены в Top-2 с confidence ≥ 0.80
- Сложные условия (8 образцов): ≥ 87% найдены в Top-5 или корректно направлены на manual review

---

## 4. Result Quality SLO (Качество результатов)

Confidence Score Distribution:
- High Confidence (> 0.95): ≥ 60% запросов
- Medium Confidence (0.80–0.95): 25–30% запросов
- Low Confidence (< 0.80): ≤ 15% запросов (требуют ручной проверки)

---

## 5. Reliability SLO (Надежность)

Error Rate: ≤ 0.1% HTTP 5xx ошибок (максимум допустимый: ≤ 1%)

---

## 6. Throughput SLI

 Количество успешно обработанных снимков в секунду (цель: обработка загружаемого изображения в реальном времени).