PYTHON ?= python3
EPOCHS ?= 30
BATCH ?= 128
DEVICE ?= auto
HF_TOKEN ?=

export HF_TOKEN

.PHONY: install prepare train evaluate predict

install:
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(PYTHON) scripts/prepare_ocr_dataset.py

train:
	$(PYTHON) scripts/train_plate_ocr.py \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH) \
		--device $(DEVICE) \
		--publish

evaluate:
	$(PYTHON) scripts/evaluate_plate_ocr.py --device $(DEVICE)

predict:
	test -n "$(SOURCE)"
	$(PYTHON) scripts/predict_plate_ocr.py --source "$(SOURCE)" --device $(DEVICE)
