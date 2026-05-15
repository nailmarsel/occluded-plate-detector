# Russian Plate Detector Training

This directory owns training and publishing the license-plate detector used by
the FastAPI app. The app expects the final Ultralytics YOLO checkpoint at:

```text
src/models/license_plate_detector.pt
```

`src/docker-compose.yml` mounts `src/models` into the app container as
`/app/models`, so this artifact is available to the app as:

```text
/app/models/license_plate_detector.pt
```

## Dataset

Use `AY000554/Car_plate_detecting_dataset` from Hugging Face. It contains
Russian car images with YOLO-format plate bounding boxes.

The training script downloads the dataset, normalizes it into this layout, and
writes a YOLO `data.yaml` file:

```text
ml/plate_detector/data/yolo/russian_plate/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
  data.yaml
```

## Local Training

From this directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

export HF_TOKEN=hf_your_token_here
make prepare

python scripts/train_plate_detector.py \
  --dataset-repo AY000554/Car_plate_detecting_dataset \
  --base-model yolo11n.pt \
  --epochs 80 \
  --imgsz 960 \
  --batch 4 \
  --device auto \
  --publish
```

Equivalent Make target:

```bash
make train
```

For training, `DEVICE=auto` resolves to CUDA when available, otherwise CPU.
Ultralytics YOLO training can hit target-assignment errors on Apple MPS, so MPS
is skipped for training. Evaluation and prediction can still use MPS through
their normal `DEVICE=auto` path. You can force a device:

```bash
make train DEVICE=cpu
```

On a MacBook M3 Pro, run a short smoke train before the full run:

```bash
make train EPOCHS=1 BATCH=2 IMGSZ=640
```

You can also pass the token only for one Make invocation:

```bash
make prepare HF_TOKEN=hf_your_token_here
```

`HF_TOKEN` is optional for public datasets, but Hugging Face applies stricter
limits to unauthenticated downloads. You can also pass the token explicitly:

```bash
python scripts/train_plate_detector.py --prepare-only --hf-token "$HF_TOKEN"
```

## Experiment Notebooks

Use notebooks for inspecting data and trying hyperparameters:

```bash
cd ml/plate_detector
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks
```

The notebooks are thin wrappers around `scripts/`, so experiments stay aligned
with Docker and repeatable training:

- `notebooks/01_dataset_check.ipynb`
- `notebooks/02_train_plate_detector.ipynb`
- `notebooks/03_evaluate_predictions.ipynb`

Use `--base-model yolo11s.pt` if you have enough GPU memory and want better
accuracy. Use `--imgsz 1280` for small or distant plates if training time is
acceptable.

## Docker Training

Build the training image:

```bash
docker build -t autobahncv-plate-trainer -f ml/plate_detector/Dockerfile .
```

Run training and publish the model into `src/models`:

```bash
docker run --rm \
  -v "$PWD/ml/plate_detector/data:/workspace/ml/plate_detector/data" \
  -v "$PWD/ml/plate_detector/runs:/workspace/ml/plate_detector/runs" \
  -v "$PWD/src/models:/workspace/src/models" \
  autobahncv-plate-trainer \
  python ml/plate_detector/scripts/train_plate_detector.py --publish
```

For NVIDIA GPU training, add Docker's GPU flag:

```bash
docker run --rm --gpus all ...
```

## Use The Model In The App

After training, verify this file exists:

```text
src/models/license_plate_detector.pt
```

Then restart the app:

```bash
cd src
docker compose --profile full up -d --build
```

Keep these environment values:

```env
NEURON2_PLATE_DETECTION_MODEL=/app/models/license_plate_detector.pt
ML_ALLOW_HEURISTIC_PLATE_FALLBACK=False
```

If search still returns `N/A`, set:

```env
ML_DEBUG_IMAGE_DIR=/tmp/autobahncv-debug
```

Then inspect the saved `*_plate_crop.jpg` files. If the crop is wrong, improve
the detector or dataset. If the crop is tight and readable, tune OCR.
