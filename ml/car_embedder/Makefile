PYTHON ?= python3
DATA_DIR ?= data/imagefolder/mad_cars
DATASET_REPO ?= yandex/mad-cars
MAX_IDENTITIES ?= 500
MAX_IMAGES_PER_IDENTITY ?= 12
EPOCHS ?= 30
BATCH ?= 16
DEVICE ?= auto
WORKERS ?= 0
HF_TOKEN ?=

export HF_TOKEN

require-data-dir = $(if $(strip $(DATA_DIR)),,$(error DATA_DIR is required. Use: make $@ DATA_DIR=/path/to/imagefolder))
require-prepared-data = $(if $(wildcard $(DATA_DIR)),$(if $(wildcard $(DATA_DIR)/*),,$(error DATA_DIR exists but has no identity folders: $(DATA_DIR). Run make prepare or pass DATA_DIR=/path/to/imagefolder)),$(error DATA_DIR does not exist: $(DATA_DIR). Run make prepare first or pass DATA_DIR=/path/to/imagefolder))

.PHONY: install prepare train evaluate export

install:
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(PYTHON) -c "import datasets, requests, tqdm" || (echo "Missing prepare dependencies. Run: make install" && exit 1)
	$(PYTHON) scripts/prepare_mad_cars_dataset.py \
		--dataset-repo $(DATASET_REPO) \
		--output-dir "$(DATA_DIR)" \
		--max-identities $(MAX_IDENTITIES) \
		--max-images-per-identity $(MAX_IMAGES_PER_IDENTITY)

train:
	$(require-data-dir)
	$(require-prepared-data)
	$(PYTHON) scripts/train_car_embedder.py \
		--data-dir "$(DATA_DIR)" \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH) \
		--device $(DEVICE) \
		--workers $(WORKERS) \
		--publish

evaluate:
	$(require-data-dir)
	$(require-prepared-data)
	$(PYTHON) scripts/evaluate_car_embedder.py \
		--data-dir "$(DATA_DIR)" \
		--batch-size $(BATCH) \
		--workers $(WORKERS) \
		--device $(DEVICE)
