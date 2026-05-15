PYTHON ?= python3
DATA ?=
BASE_MODEL ?= yolo11n.pt
EPOCHS ?= 80
IMGSZ ?= 960
BATCH ?= 16
DEVICE ?= auto
HF_TOKEN ?=
RUN_NAME ?= car_detector_yolo
BEST ?= runs/$(RUN_NAME)/weights/best.pt
ARTIFACT ?= ../../src/models/car_detector.pt

export HF_TOKEN

require-best = $(if $(wildcard $(BEST)),,$(error Missing trained checkpoint: $(BEST). Run make train DATA=/path/to/data.yaml first or set BEST=/path/to/best.pt))

.PHONY: install train publish evaluate predict

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	test -n "$(DATA)"
	$(PYTHON) scripts/train_car_detector.py \
		--data "$(DATA)" \
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
	test -n "$(DATA)"
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
