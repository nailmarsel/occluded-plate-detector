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

```bash
cd ml/car_embedder
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python scripts/train_car_embedder.py \
  --data-dir /path/to/imagefolder \
  --epochs 30 \
  --publish
```

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
