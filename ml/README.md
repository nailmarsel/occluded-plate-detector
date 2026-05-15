# ML Workspaces

This directory separates model development from the FastAPI application.

| Workspace        | App neuron                | Artifact                               |
|------------------|---------------------------|----------------------------------------|
| `car_detector`   | Neuron 1: car detection   | `src/models/car_detector.pt`           |
| `plate_detector` | Neuron 2: plate detection | `src/models/license_plate_detector.pt` |
| `plate_ocr`      | Neuron 3: plate OCR       | `src/models/plate_ocr.pt`              |
| `car_embedder`   | Neuron 4: image embedding | `src/models/car_embedder.pt`           |

The Docker app mounts `src/models` into `/app/models`, so published artifacts are
available to the app container.

The app loads these files from `src/models` when the matching environment
variables point to `/app/models/*.pt` inside Docker:

```env
NEURON1_CAR_DETECTION_MODEL=/app/models/car_detector.pt
NEURON2_PLATE_DETECTION_MODEL=/app/models/license_plate_detector.pt
NEURON3_OCR_MODEL=/app/models/plate_ocr.pt
NEURON4_RESNET_MODEL=/app/models/car_embedder.pt
```

If a training run was completed without `--publish`, copy its `best.pt` from
the workspace `runs/` directory to the artifact name shown in the table.

Monitoring, drift checks, feedback collection, registry fields, and retraining
rules for all four models are described in
[`../specs/Monitoring_Retraining.md`](../specs/Monitoring_Retraining.md).
