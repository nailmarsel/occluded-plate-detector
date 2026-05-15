FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ml/plate_detector/requirements.txt /tmp/plate-detector-requirements.txt
RUN pip install --no-cache-dir -r /tmp/plate-detector-requirements.txt

COPY ml/plate_detector /workspace/ml/plate_detector

CMD ["python", "ml/plate_detector/scripts/train_plate_detector.py", "--publish"]
