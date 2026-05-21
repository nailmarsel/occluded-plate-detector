## Readiness

| Блок | Готово? | Где артефакт | Комментарий |
|---|---|---|---|
| **Requirements / acceptance criteria** | Да | [`/specs/`](https://github.com/nailmarsel/occluded-plate-detector/tree/main/specs) (PRD.md, DoD.md) | Критерии описаны в файлах |
| **Data / dataset quality** | Да | [`/specs/Data_Spec.md`](https://github.com/nailmarsel/occluded-plate-detector/tree/main/specs) | Подробный Data Spec по двум датасетам Hugging Face (`AY000554/Car_plate_detecting_dataset`, `AY000554/Car_plate_OCR_dataset`). Описаны источники, разбиение train/val/test, формат разметки, и три блока валидаций. |
| **Experiments / baseline** | Да | [`/notebooks/`](https://github.com/nailmarsel/occluded-plate-detector/tree/main/notebooks), [`/ml/`](https://github.com/nailmarsel/occluded-plate-detector/tree/main/ml), [`/reports/models/`](https://github.com/nailmarsel/occluded-plate-detector/tree/main/reports/models) | В notebooks лежат файлы с тестами моделей. В ml скрипты для тренировки и обучения. В /reports/models описано правильное использование моделей и весов. |
| **Deployment** | Да | [`/src/`](https://github.com/nailmarsel/occluded-plate-detector/tree/main/src), [`/.github/workflows/`](https://github.com/nailmarsel/occluded-plate-detector/tree/main/.github/workflows) | В src лежит код приложения, конфигурационные файлы и docker-compose.yaml + Dockerfile для разворота. В /.github/workflows/ конфигурация для CI/CD  |
| **Monitoring / retraining** | Частично | README §5 «Observability & Monitoring», схема observability | План полностью описан: `prometheus-fastapi-instrumentator` + `/metrics`, созданы дашборды в Grafana, индексы Elasticsearch, алерты, retraining недоделан. Эндпоинты Grafana и Kibana - /grafana и /kibana
