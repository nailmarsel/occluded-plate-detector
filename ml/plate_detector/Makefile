PYTHON ?= python3
DATASET_REPO ?= AY000554/Car_plate_detecting_dataset
BASE_MODEL ?= yolo11n.pt
EPOCHS ?= 80
IMGSZ ?= 960
BATCH ?= 16
DEVICE ?= auto
HF_TOKEN ?=

export HF_TOKEN

.PHONY: install prepare train evaluate predict

install:
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(PYTHON) scripts/train_plate_detector.py \
		--dataset-repo $(DATASET_REPO) \
		--prepare-only

train:
	$(PYTHON) scripts/train_plate_detector.py \
		--dataset-repo $(DATASET_REPO) \
		--base-model $(BASE_MODEL) \
		--epochs $(EPOCHS) \
		--imgsz $(IMGSZ) \
		--batch $(BATCH) \
		--device $(DEVICE) \
		--publish

evaluate:
	$(PYTHON) scripts/evaluate_plate_detector.py \
		--imgsz $(IMGSZ) \
		--device $(DEVICE)

predict:
	test -n "$(SOURCE)"
	$(PYTHON) scripts/predict_plate_detector.py \
		--source "$(SOURCE)" \
		--imgsz $(IMGSZ) \
		--device $(DEVICE)
