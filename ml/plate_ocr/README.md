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

On a MacBook M3 Pro, the Makefile defaults to `BATCH=64 WORKERS=0`. Run a short
smoke train before a full run:

```bash
make train EPOCHS=1 BATCH=32
```

Published artifact:

```text
src/models/plate_ocr.pt
```

The app loads this custom checkpoint from `src/models/plate_ocr.pt` when
`NEURON3_OCR_MODEL=/app/models/plate_ocr.pt` is set in Docker.

## Notebooks

```bash
jupyter lab notebooks
```
