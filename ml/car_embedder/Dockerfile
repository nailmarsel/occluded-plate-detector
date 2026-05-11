FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY ml/car_embedder/requirements.txt /tmp/car-embedder-requirements.txt
RUN pip install --no-cache-dir -r /tmp/car-embedder-requirements.txt

COPY ml/car_embedder /workspace/ml/car_embedder

CMD ["python", "ml/car_embedder/scripts/train_car_embedder.py", "--help"]
