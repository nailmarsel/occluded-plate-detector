PYTHON ?= python3
DATA_DIR ?=
EPOCHS ?= 30
BATCH ?= 64
DEVICE ?= auto
HF_TOKEN ?=

export HF_TOKEN

.PHONY: install train evaluate export

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	test -n "$(DATA_DIR)"
	$(PYTHON) scripts/train_car_embedder.py \
		--data-dir "$(DATA_DIR)" \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH) \
		--device $(DEVICE) \
		--publish

evaluate:
	test -n "$(DATA_DIR)"
	$(PYTHON) scripts/evaluate_car_embedder.py \
		--data-dir "$(DATA_DIR)" \
		--device $(DEVICE)
