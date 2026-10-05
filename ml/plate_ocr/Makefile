PYTHON ?= python3
EPOCHS ?= 30
BATCH ?= 64
DEVICE ?= cpu
WORKERS ?= 0
HF_TOKEN ?=
FT_EPOCHS ?= 8
FT_LR ?= 0.0001
MAX_MINUTES ?= 25
LIMIT ?= 0

export HF_TOKEN

.PHONY: install prepare train evaluate predict finetune compare

install:
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(PYTHON) scripts/prepare_ocr_dataset.py

train:
	$(PYTHON) scripts/train_plate_ocr.py \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH) \
		--device $(DEVICE) \
		--workers $(WORKERS) \
		--publish

evaluate:
	$(PYTHON) scripts/evaluate_plate_ocr.py --batch-size $(BATCH) --device $(DEVICE) --workers $(WORKERS)

predict:
	test -n "$(SOURCE)"
	$(PYTHON) scripts/predict_plate_ocr.py --source "$(SOURCE)" --device $(DEVICE)

# Fine-tune the published OCR (src/models/plate_ocr.pt).
finetune:
	$(PYTHON) scripts/finetune_plate_ocr.py \
		--epochs $(FT_EPOCHS) \
		--lr $(FT_LR) \
		--batch-size $(BATCH) \
		--max-minutes $(MAX_MINUTES) \
		--device $(DEVICE) \
		--workers $(WORKERS)

# Old vs fine-tuned on clean and degraded test -> reports/models/finetune/plate_ocr.json
compare:
	$(PYTHON) scripts/compare_plate_ocr.py \
		--batch-size $(BATCH) \
		--device $(DEVICE) \
		--workers $(WORKERS) \
		--limit $(LIMIT)
