PYTHON ?= python3
DATASET_REPO ?= AY000554/Car_plate_detecting_dataset
BASE_MODEL ?= yolo11n.pt
EPOCHS ?= 80
IMGSZ ?= 960
BATCH ?= 4
DEVICE ?= auto
WORKERS ?= 0
HF_TOKEN ?=
RUN_NAME ?= russian_plate_yolo
BEST ?= runs/$(RUN_NAME)/weights/best.pt
ARTIFACT ?= ../../src/models/license_plate_detector.pt
FT_EPOCHS ?= 10
FT_LR0 ?= 0.001
FRACTION ?= 1.0
OCCLUSION ?= 0.3
MAX_MINUTES ?= 25
LIMIT ?= 0

export HF_TOKEN

require-best = $(if $(wildcard $(BEST)),,$(error Missing trained checkpoint: $(BEST). Run make train first or set BEST=/path/to/best.pt))

.PHONY: install prepare train publish evaluate predict finetune compare

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
		--workers $(WORKERS) \
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

# Fine-tune the published detector (src/models/license_plate_detector.pt).
finetune:
	$(PYTHON) scripts/finetune_plate_detector.py \
		--epochs $(FT_EPOCHS) \
		--lr0 $(FT_LR0) \
		--imgsz $(IMGSZ) \
		--batch $(BATCH) \
		--fraction $(FRACTION) \
		--occlusion-fraction $(OCCLUSION) \
		--max-minutes $(MAX_MINUTES) \
		--device $(DEVICE) \
		--workers $(WORKERS)

# Old vs fine-tuned on clean and occluded test -> reports/models/finetune/plate_detector.json
compare:
	$(PYTHON) scripts/compare_plate_detectors.py \
		--imgsz $(IMGSZ) \
		--device $(DEVICE) \
		--limit $(LIMIT)
