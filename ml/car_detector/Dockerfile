FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ml/car_detector/requirements.txt /tmp/car-detector-requirements.txt
RUN pip install --no-cache-dir -r /tmp/car-detector-requirements.txt

COPY ml/car_detector /workspace/ml/car_detector

CMD ["python", "ml/car_detector/scripts/train_car_detector.py", "--help"]
