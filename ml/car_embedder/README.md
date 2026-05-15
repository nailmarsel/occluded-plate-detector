# Car Embedder Training

This workspace trains a vehicle embedding backbone for image similarity search.

The app currently uses ImageNet-pretrained TorchVision ResNet features. That is
a reasonable baseline, but a domain-trained embedder can improve retrieval when
you have identity labels, for example one folder per car/plate:

```text
data/imagefolder/
  A864AA199/
    img1.jpg
    img2.jpg
  E507MP136/
    img1.jpg
```

The training script fine-tunes a ResNet classifier on those identities and
publishes the backbone weights for future app integration.

## Local Training

This module can prepare a bounded subset of
[`yandex/mad-cars`](https://huggingface.co/datasets/yandex/mad-cars) from
Hugging Face. That dataset has a `car_id` column, so it can be converted into an
ImageFolder dataset where every subdirectory is one vehicle identity.

`AY000554/Car_plate_detecting_dataset` is not used for the embedder because it
contains plate bounding boxes, not vehicle identity labels. It is useful for
plate detection, but it cannot tell the embedder which images show the same car.

```bash
cd ml/car_embedder
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

make prepare
make train
```

If `make prepare` reports `No module named 'datasets'`, reinstall the updated
requirements inside this venv:

```bash
make install
```

By default, `make prepare` writes:

```text
ml/car_embedder/data/imagefolder/mad_cars/
  0/
    0_000.jpg
    1_001.jpg
  1/
    0_000.jpg
```

You can tune the download size:

```bash
make prepare MAX_IDENTITIES=1000 MAX_IMAGES_PER_IDENTITY=16
```

Or train with your own ImageFolder dataset:

```bash
python scripts/train_car_embedder.py \
  --data-dir /path/to/imagefolder \
  --epochs 30 \
  --device auto \
  --publish
```

Equivalent Make target:

```bash
make train DATA_DIR=/path/to/imagefolder
```

Running `make train` without `DATA_DIR` uses the default prepared MAD Cars
folder: `data/imagefolder/mad_cars`.

`--device auto` resolves to CUDA when available, then Apple MPS, then CPU. You
can force a device with `DEVICE=cpu`, `DEVICE=cuda`, or `DEVICE=mps`.

Published artifact:

```text
src/models/car_embedder.pt
```

The current app does not yet load this custom checkpoint; it keeps using
TorchVision ResNet by name. This workspace prepares the training process so the
app backend can be switched cleanly when you have enough identity data.

## Notebooks

```bash
jupyter lab notebooks
```
