# Car Detector Training

This workspace is for fine-tuning the vehicle detector used by Neuron 1.

The app already supports any Ultralytics YOLO checkpoint through:

```env
NEURON1_CAR_DETECTION_MODEL=/app/models/car_detector.pt
```

The default app uses a COCO-pretrained model and filters vehicle classes. Train a
custom detector only if your images are consistently missed, cropped badly, or
come from a very specific camera/domain.

If you do not have a custom car dataset yet, publish the COCO-pretrained YOLO
base model as the app's car detector:

```bash
make publish-base
```

That creates:

```text
src/models/car_detector.pt
```

The app will still filter YOLO COCO vehicle classes for this base detector.

## Dataset

By default, `make prepare` copies the already downloaded
`../plate_detector/data/raw/AY000554__Car_plate_detecting_dataset` dataset into
`ml/car_detector/data/raw/` and works from that local copy. Its labels are
license-plate boxes, not car boxes, so the car detector prepare step does
**not** reuse those labels. Instead it runs the COCO-pretrained YOLO vehicle
detector over the images and writes pseudo-labels for the best car-like object
in each image.

This gives a same-style local workflow and a car-image domain adaptation
dataset, but it is weaker than a real manually labeled whole-car detection
dataset. If you have a proper YOLO car dataset, pass its `data.yaml` directly.

## Local Training

```bash
cd ml/car_detector
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

make prepare
make train
```

If `make prepare` reports `No module named 'huggingface_hub'`, reinstall the
updated requirements inside this venv:

```bash
make install
```

`make prepare` writes:

```text
ml/car_detector/data/yolo/car_from_plate_dataset/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
  data.yaml
```

For a quick smoke test of preparation, limit each split:

```bash
make prepare MAX_IMAGES=100
```

To copy from a different existing local dataset:

```bash
make prepare SOURCE_RAW_DIR=/path/to/AY000554__Car_plate_detecting_dataset
```

To train on your own YOLO car dataset instead:

```bash
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

Without `DATA=...`, `make train` uses the prepared default:

```text
data/yolo/car_from_plate_dataset/data.yaml
```

`make train` publishes the trained checkpoint automatically to
`src/models/car_detector.pt`. If training already finished and you only need to
copy the checkpoint again, run:

```bash
make publish
```

For training, `--device auto` resolves to CUDA when available, otherwise CPU.
Ultralytics YOLO training can hit target-assignment errors on Apple MPS, so MPS
is skipped for training. You can force a device with `DEVICE=cpu` or `DEVICE=0`.

On a MacBook M3 Pro, the default `BATCH=4 WORKERS=0` is intentionally
conservative. Run a short smoke train before a full run:

```bash
make train EPOCHS=1 BATCH=2 IMGSZ=640
```

The published artifact is:

```text
src/models/car_detector.pt
```

## Notebooks

```bash
jupyter lab notebooks
```
