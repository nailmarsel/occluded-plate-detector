PYTHON ?= python3
DATA ?=
BASE_MODEL ?= yolo11n.pt
EPOCHS ?= 80
IMGSZ ?= 960
BATCH ?= 16
DEVICE ?= auto
HF_TOKEN ?=

export HF_TOKEN

.PHONY: install train evaluate predict

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
		--publish

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
