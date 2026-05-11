FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY ml/plate_ocr/requirements.txt /tmp/plate-ocr-requirements.txt
RUN pip install --no-cache-dir -r /tmp/plate-ocr-requirements.txt

COPY ml/plate_ocr /workspace/ml/plate_ocr

CMD ["python", "ml/plate_ocr/scripts/train_plate_ocr.py", "--help"]
