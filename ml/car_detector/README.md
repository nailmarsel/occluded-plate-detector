# Car Detector Training

This workspace is for fine-tuning the vehicle detector used by Neuron 1.

The app already supports any Ultralytics YOLO checkpoint through:

```env
NEURON1_CAR_DETECTION_MODEL=/app/models/car_detector.pt
```

The default app uses a COCO-pretrained model and filters vehicle classes. Train a
custom detector only if your images are consistently missed, cropped badly, or
come from a very specific camera/domain.

## Expected Dataset

Provide any YOLO detection dataset with a `data.yaml`. If you train a single
class detector, use class `0: car`. If you train COCO-style classes, keep vehicle
class ids compatible with the app or adjust `VEHICLE_CLASS_IDS`.

## Local Training

```bash
cd ml/car_detector
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python scripts/train_car_detector.py \
  --data /path/to/data.yaml \
  --base-model yolo11n.pt \
  --epochs 80 \
  --imgsz 960 \
  --device auto \
  --publish
```

Equivalent Make target:

```bash
make train DATA=/path/to/data.yaml
```

`--device auto` resolves to CUDA when available, then Apple MPS, then CPU. You
can force a device with `DEVICE=cpu` or `DEVICE=0`.

The published artifact is:

```text
src/models/car_detector.pt
```

## Notebooks

```bash
jupyter lab notebooks
```
