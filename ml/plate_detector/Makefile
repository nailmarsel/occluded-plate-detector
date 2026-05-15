PYTHON ?= python3
DATASET_REPO ?= AY000554/Car_plate_detecting_dataset
BASE_MODEL ?= yolo11n.pt
EPOCHS ?= 3
IMGSZ ?= 960
BATCH ?= 16
DEVICE ?= auto
HF_TOKEN ?=
RUN_NAME ?= russian_plate_yolo
BEST ?= runs/$(RUN_NAME)/weights/best.pt
ARTIFACT ?= ../../src/models/license_plate_detector.pt

export HF_TOKEN

require-best = $(if $(wildcard $(BEST)),,$(error Missing trained checkpoint: $(BEST). Run make train first or set BEST=/path/to/best.pt))

.PHONY: install prepare train publish evaluate predict

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
		--name $(RUN_NAME) \
		--publish

publish:
	$(require-best)
	mkdir -p "$$(dirname "$(ARTIFACT)")"
	cp "$(BEST)" "$(ARTIFACT)"
	@echo "Published detector: $(ARTIFACT)"

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
