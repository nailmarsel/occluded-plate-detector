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

The app currently uses:

- custom YOLO path for car/plate detectors,
- EasyOCR for OCR,
- TorchVision ResNet for embeddings.

The OCR and embedder workspaces prepare trainable replacements; app integration
can be switched once their validation metrics are good enough.

Monitoring, drift checks, feedback collection, registry fields, and retraining
rules for all four models are described in
[`../specs/Monitoring_Retraining.md`](../specs/Monitoring_Retraining.md).
