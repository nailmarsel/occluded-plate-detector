PYTHON ?= python3
DATA ?= data/yolo/car_from_plate_dataset/data.yaml
DATASET_REPO ?= AY000554/Car_plate_detecting_dataset
SOURCE_RAW_DIR ?= ../plate_detector/data/raw/AY000554__Car_plate_detecting_dataset
RAW_DATASET_DIR ?= data/raw/AY000554__Car_plate_detecting_dataset
BASE_MODEL ?= yolo11n.pt
EPOCHS ?= 80
IMGSZ ?= 960
BATCH ?= 4
DEVICE ?= auto
WORKERS ?= 0
HF_TOKEN ?=
MAX_IMAGES ?= 0
RUN_NAME ?= car_detector_yolo
BEST ?= runs/$(RUN_NAME)/weights/best.pt
ARTIFACT ?= ../../src/models/car_detector.pt

export HF_TOKEN

require-data = $(if $(strip $(DATA)),,$(error DATA is required. Use: make train DATA=/path/to/data.yaml))
require-data-file = $(if $(wildcard $(DATA)),,$(error Missing dataset: $(DATA). Run make prepare first or set DATA=/path/to/data.yaml))
require-source-raw = $(if $(wildcard $(SOURCE_RAW_DIR)),,$(error Missing source dataset: $(SOURCE_RAW_DIR). Run ml/plate_detector make prepare first or set SOURCE_RAW_DIR=/path/to/dataset))
require-best = $(if $(wildcard $(BEST)),,$(error Missing trained checkpoint: $(BEST). Run make train DATA=/path/to/data.yaml first or set BEST=/path/to/best.pt))
require-base = $(if $(wildcard $(BASE_MODEL)),,$(error Missing base model: $(BASE_MODEL). Run make train once to download it, or set BASE_MODEL=/path/to/yolo.pt))

.PHONY: install prepare train publish publish-base evaluate predict

install:
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(require-source-raw)
	$(PYTHON) -c "import PIL, tqdm, ultralytics" || (echo "Missing prepare dependencies. Run: make install" && exit 1)
	mkdir -p "$$(dirname "$(RAW_DATASET_DIR)")"
	test -d "$(RAW_DATASET_DIR)/extracted/train/images" || (rm -rf "$(RAW_DATASET_DIR)" && cp -R "$(SOURCE_RAW_DIR)" "$(RAW_DATASET_DIR)")
	$(PYTHON) scripts/prepare_car_detector_dataset.py \
		--dataset-repo $(DATASET_REPO) \
		--source-raw-dir "$(RAW_DATASET_DIR)" \
		--base-model $(BASE_MODEL) \
		--imgsz $(IMGSZ) \
		--device $(DEVICE) \
		--max-images $(MAX_IMAGES)

train:
	$(require-data)
	$(require-data-file)
	$(PYTHON) scripts/train_car_detector.py \
		--data "$(DATA)" \
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

publish-base:
	$(require-base)
	mkdir -p "$$(dirname "$(ARTIFACT)")"
	cp "$(BASE_MODEL)" "$(ARTIFACT)"
	@echo "Published base detector: $(ARTIFACT)"

evaluate:
	$(require-data)
	$(require-data-file)
	$(PYTHON) scripts/evaluate_car_detector.py \
		--data "$(DATA)" \
		--imgsz $(IMGSZ) \
		--device $(DEVICE)

predict:
	test -n "$(SOURCE)"
	$(PYTHON) scripts/predict_car_detector.py \
		--source "$(SOURCE)" \
		--imgsz $(IMGSZ) \
		--device $(DEVICE)
