# Plate OCR Training

This workspace trains a plate OCR model from cropped Russian plate images.

Dataset:

```text
AY000554/Car_plate_OCR_dataset
```

The dataset labels are inferred from image filenames, for example:

```text
A129XY196.png -> A129XY196
```

The scripts train a compact CNN + BiLSTM + CTC recognizer. It is useful for
experiments and future replacement of EasyOCR in the app.

## Local Training

```bash
cd ml/plate_ocr
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_your_token_here
make prepare
make train
```

`HF_TOKEN` is optional for public datasets, but it enables authenticated
Hugging Face requests with higher rate limits.

You can also pass the token only for one Make invocation:

```bash
make prepare HF_TOKEN=hf_your_token_here
```

OCR training uses CTC loss, which PyTorch does not currently implement on Apple
MPS. The Makefile therefore defaults to CPU. `DEVICE=auto` resolves to CUDA
when available, otherwise CPU. You can force a device for training or
evaluation:

```bash
make train DEVICE=cpu
```

Published artifact:

```text
src/models/plate_ocr.pt
```

The current app still uses EasyOCR. This model workspace is prepared so we can
swap the app OCR backend later without changing the training process.

## Notebooks

```bash
jupyter lab notebooks
```
