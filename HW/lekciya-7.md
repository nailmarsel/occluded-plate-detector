# Monitoring & Observability – AutobahnCV

Документ описывает реализацию наблюдаемости в проекте согласно чек-листу.

## 1. Сервисный мониторинг

- **Метрики:**  
  - `http_requests_total` – количество запросов  
  - `http_request_duration_seconds` – гистограмма задержек  
  - `http_requests_total{status=~"5.."}` – ошибки  
- **Расчётные показатели:**  
  - **Throughput** = `rate(http_requests_total[1m])`  
  - **Error Rate** = `rate(http_requests_total{status=~"5.."}[1m]) / rate(http_requests_total[1m])`  
  - **p95 Latency** = `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))`  
  - **Availability** = `1 - (rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]))`  
- **Где смотреть:** Grafana дашборд «Service Metrics»  

## 2. Мониторинг входных данных

- **Метрики (Prometheus):**  
  - `autobahncv_images_processed_total` – всего обработано изображений  
  - `autobahncv_input_errors_total` – ошибки валидации (формат, пустой файл)  
  - `autobahncv_confidence_car`, `autobahncv_confidence_plate`, `autobahncv_confidence_ocr` – уверенность детекторов  
  - `autobahncv_plate_length` – длина распознанного номера  
  - `autobahncv_image_size_bytes` – размер загружаемых изображений  
- **Логирование в Elasticsearch:** каждый инференс сохраняется в индексе `inference-logs` (поля: confidence, plate_length, image_size_bytes, success, ошибки)  
- **Что отслеживается:** пропуски, пустые значения, некорректные входы, изменение распределений (drift)  

## 3. Мониторинг предсказаний модели

- **Метрики:**  
  - `autobahncv_plate_fallback_total` – доля эвристического определения номера  
  - `autobahncv_search_similarity_score` – гистограмма scores поиска  
  - Доля low‑confidence предсказаний (car confidence < 0.6)  
- **Дашборд:** «Model Predictions» отображает fallback rate, средний similarity score, heatmap распределения  

## 4. Мониторинг бизнес-поведения

- **API обратной связи:** `POST /api/v1/feedback`  
  - Действия: `confirm`, `reject`, `correct`  
  - Флаг `disputed` для спорных кейсов  
- **Метрика:** `autobahncv_feedback_actions_total` с лейблом `action`  
- **Логирование:** индекс `feedback-logs` в Elasticsearch  
- **Дашборд:** «Business Feedback» показывает количество действий и confirmation rate  

## 5. Инструменты observability

| Задача | Инструмент |
|--------|------------|
| Сбор метрик | `prometheus-client` + `/metrics` endpoint |
| Хранение / агрегация | Prometheus |
| Дашборды | Grafana |
| Drift / profiling | Prometheus-алерты + Elasticsearch (`inference-logs`) |
| Версионирование моделей | MLflow Model Registry |
| Audit / retraining pipeline | Ручной (на основе алертов), перспектива – CI/CD |

## 6. Дашборды (Grafana)

1. **Service Metrics** – Throughput, Error Rate, p95 Latency, Availability (фильтр по `handler`)  
2. **Input Data** – confidence, длина номера, размер изображения, ошибки валидации  
3. **Model Predictions** – fallback rate, similarity score, heatmap  
4. **Quality & Drift** – динамика confidence, low‑confidence, fallback, ошибки  

## 7. Алерты

| Алерт | Условие |
|-------|---------|
| High Error Rate | `error_rate > 0.1` (10%) за 5 мин |
| High p95 Latency | `p95 > 2` сек за 5 мин |
| Drift (confidence drop) | средний confidence car < 0.6 за 10 мин |
| High Fallback Rate | `fallback_rate > 0.5` за 10 мин |
| High Low‑Confidence | `low_confidence_rate > 30%` за 10 мин |
| Audit / retraining pipeline | вручную (контроль логов CI/CD) |

## 8. Human Feedback Loop

- **Эндпоинт:** `POST /api/v1/feedback`  
- **Действия:** подтверждение, исправление, отклонение  
- **Сохранение:** Elasticsearch индекс `feedback-logs` + Prometheus метрика  
- **Трудные кейсы:** накопление `disputed` и `rejected` для анализа и дообучения  

## 9. Versioning / Model Registry

- **Текущая основная модель** – версия в стадии `Production` в MLflow Registry  
- **Кандидат** – версия в `Staging`  
- **Данные:** параметр `dataset` и теги в MLflow  
- **Метрики:** зафиксированы в MLflow для каждой версии  
- **Отличия:** сравнение метрик в MLflow UI, описание в `CHANGELOG.md`  

## 10. Retraining Policy

### Когда запускать аудит
- Срабатывание алертов (Error Rate, Fallback, Confidence Drop)  
- Доля low‑confidence > 30% в течение 1 часа  
- Накопление > 50 `disputed` кейсов  
- Плановый ежемесячный аудит  

### Когда запускать переобучение
- Падение метрик (precision, recall, top‑1 accuracy) > 10%  
- Появление новых данных, статистически улучшающих метрики на валидации  
- Накопление достаточного количества исправленных примеров  

### Когда переобучение не нужно
- Метрики стабильны, алерты не срабатывают  
- Нет новых данных  
- Кандидат не показывает улучшений  

### Кто принимает решение
- ML‑инженер (ручное подтверждение)  

### Как сравниваются модели
- По метрикам в MLflow на фиксированной тестовой выборке  

### Как модель попадает в rollout
1. Кандидат → `Staging` в MLflow  
2. A/B‑тестирование или ручная оценка  
3. При успехе → `Production`, обновление `model_version.txt` и Docker‑образа  
4. Перезапуск сервиса  

---

**Все компоненты мониторинга развёрнуты, описаны и проверены.**